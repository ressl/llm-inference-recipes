import argparse, json, time, urllib.request
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--url', default='http://127.0.0.1:30000')
p.add_argument('--output', required=True)
a=p.parse_args()
model='deepseek-ai/DeepSeek-V4.1-Flash'
results=[]

def record(name, data):
    results.append({'name':name, **data})
    Path(a.output).write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print(name, json.dumps(data,ensure_ascii=False), flush=True)

def chat(name, messages, **extra):
    body={'model':model,'messages':messages,'temperature':0,'max_tokens':512,
          'chat_template_kwargs':{'enable_thinking':False},**extra}
    req=urllib.request.Request(a.url+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    t=time.monotonic()
    with urllib.request.urlopen(req,timeout=600) as r: result=json.load(r)
    record(name,{'seconds':time.monotonic()-t,'response':result})
    assert result['model']==model
    return result['choices'][0]

user=lambda s:[{'role':'user','content':s}]
basic=chat('arithmetic',user('Antworte ausschließlich mit der Zahl: Was ist 17 mal 19?'))
assert basic['message']['content'].strip()=='323',basic
assert basic['finish_reason']=='stop'

reason=chat('reasoning',user('Löse sorgfältig: Ein Zug fährt 90 km/h. Wie viele Kilometer fährt er in 40 Minuten? Antworte abschließend mit der Zahl und Einheit.'),chat_template_kwargs={'enable_thinking':True,'reasoning_effort':1},max_tokens=2048)
assert '60' in reason['message']['content'],reason
assert reason['message'].get('reasoning') or reason['message'].get('reasoning_content'),reason

tools=[{'type':'function','function':{'name':'get_room_temperature','description':'Liest die Temperatur eines benannten Raumes.','parameters':{'type':'object','properties':{'room':{'type':'string'}},'required':['room'],'additionalProperties':False}}}]
messages=user('Lies mit dem Werkzeug die Temperatur im Serverraum. Verwende room="Serverraum".')
call=chat('tool_call',messages,tools=tools,tool_choice='auto')
assert call['finish_reason']=='tool_calls',call
tc=call['message']['tool_calls'][0]
assert tc['function']['name']=='get_room_temperature',tc
assert json.loads(tc['function']['arguments'])=={'room':'Serverraum'},tc
messages.append(call['message'])
messages.append({'role':'tool','tool_call_id':tc['id'],'content':'{"room":"Serverraum","temperature_celsius":21.7}'})
answer=chat('tool_result',messages,tools=tools,tool_choice='auto')
assert any(x in answer['message']['content'] for x in ('21.7','21,7')),answer

body={'model':model,'messages':user('Antworte genau mit: STREAM_OK'),'max_tokens':64,'temperature':0,'stream':True,'chat_template_kwargs':{'enable_thinking':False}}
req=urllib.request.Request(a.url+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
chunks=[]; content='';done=False;t=time.monotonic();first=None
with urllib.request.urlopen(req,timeout=600) as r:
    for raw in r:
        if not raw.startswith(b'data: '):continue
        text=raw[6:].decode().strip()
        if text=='[DONE]':done=True;break
        chunk=json.loads(text);chunks.append(chunk)
        for c in chunk.get('choices',[]):
            token=c.get('delta',{}).get('content') or ''
            if token and first is None:first=time.monotonic()-t
            content+=token
record('stream',{'seconds':time.monotonic()-t,'first_content_seconds':first,'content':content,'done':done,'chunks':chunks})
assert done and content.strip()=='STREAM_OK',(done,content)
print('Basic generation, reasoning, tool round trip and streaming passed',flush=True)
