import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from soak_conversations import consume_chat


def event(delta, finish=None, usage=None):
    return b'data: ' + json.dumps({'choices': [{'delta': delta, 'finish_reason': finish}],
                                  'usage': usage}).encode() + b'\n'


class ChatStreamTests(unittest.TestCase):
    def test_fragmented_tool_arguments_and_usage_only_chunk(self):
        lines = [event({'tool_calls': [{'index': 0, 'id': 'call_1', 'function':
                  {'name': 'get_test_report', 'arguments': '{"case'}}]}),
                 event({'tool_calls': [{'index': 0, 'function': {'arguments': '_id":"a"}'}}]},
                       'tool_calls'),
                 b'data: {"choices":[],"usage":{"prompt_tokens":123,"completion_tokens":9}}\n',
                 b'data: [DONE]\n']
        message, info = consume_chat(lines, clock=lambda: 42)
        call = message['tool_calls'][0]
        self.assertEqual({'case_id': 'a'}, json.loads(call['function']['arguments']))
        self.assertEqual('get_test_report', call['function']['name'])
        self.assertEqual('call_1', call['id'])
        self.assertEqual(42, info['first_token_time'])
        self.assertEqual(9, info['usage']['completion_tokens'])

    def test_reasoning_and_content_preserved_separately(self):
        message, info = consume_chat([event({'reasoning': 'Check.'}), event({'content': 'OK'},
            'stop', {'prompt_tokens': 12, 'completion_tokens': 4}), b'data: [DONE]\n'])
        self.assertEqual('OK', message['content'])
        self.assertEqual('Check.', message['reasoning'])
        self.assertEqual('stop', info['finish_reason'])

    def test_disconnect_error_missing_usage_or_missing_content_rejected(self):
        valid = event({'content': 'OK'}, 'stop', {'prompt_tokens': 10, 'completion_tokens': 1})
        for lines in [[valid], [b'data: {"error":{"message":"failed"}}\n'],
                      [event({'content': 'OK'}, 'stop'), b'data: [DONE]\n'],
                      [event({}, 'stop', {'prompt_tokens': 10, 'completion_tokens': 1}),
                       b'data: [DONE]\n']]:
            with self.subTest(lines=lines), self.assertRaises(ValueError):
                consume_chat(lines)


if __name__ == '__main__':
    unittest.main()
