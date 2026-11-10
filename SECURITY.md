# Security policy

## Intended use

`expanded` is a static reverse-engineering toolset for research and defensive
analysis. It is intended exclusively for:

- Defensive malware analysis in isolated laboratory VMs
- Academic and classroom instruction in reverse engineering
- Incident-response work where the analyst has lawful possession of the sample
- Openly-published security research

It must **not** be used:

- Against software you do not own or lack written permission to analyse
- To defeat licensing or DRM on commercial software
- To distribute unpacked malware binaries to parties who would not otherwise have
  lawful access

Nothing in this repository is a grant of permission to analyse a particular
binary. Lawful possession and, where it applies, written authorisation are the
user's responsibility. The tools are static and never execute the protected
sample natively; the only component that runs code is a contained software
emulator whose output is labeled *observed*.

## Reporting a vulnerability in the tooling

If you find a security issue in this software itself — for example a crafted
input that causes memory-unsafe behaviour in a parser, or a sandbox-escape in the
contained emulator — please report it privately rather than opening a public
issue:

1. Open a [GitHub security advisory](https://github.com/Vihaan2012-cmyk/vmpx-expanded/security/advisories/new)
   (preferred), or
2. Contact the maintainer through their GitHub profile.

Please include a minimal reproducer and the affected crate/version. Give a
reasonable window for a fix before public disclosure. Do **not** attach live
malware samples; a description or a benign reproducer is sufficient.

## Supported versions

This is a research project; fixes land on the default branch. There is no
long-term support branch — build from the latest commit.
