# Security

Use GitHub's private vulnerability reporting for security issues. Include the
affected revision and a minimal reproduction without credentials or private
data. General setup and performance questions can use public issues.

The example API binds to the host's loopback interface. Deployments that expose
it to other machines need their own authentication, TLS and access controls.
The recipe does not provide an authenticated public service.

Immutable dependency pins make a measurement reproducible; they do not imply
that an old image remains free of vulnerabilities. Review upstream advisories
and requalify an updated runtime before changing the pins. This initial recipe
uses preview runtime code and local patches; it is not a supported upstream
release. GPU containers require access to the selected devices.
