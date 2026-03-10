# Introduction to Flox

**Duration**: 3-5 minutes | **GPU required**: No

## What This Demo Shows

Flox creates declarative, reproducible development environments. You declare what you need in a manifest, Flox handles the rest, and it works the same everywhere.

## The Problem

Setting up development environments today means:
- README files with 50 manual steps
- "Works on my machine" syndrome
- Environment drift between dev, staging, and production
- Onboarding new developers takes hours or days
- Documentation that's always out of date

## Demo Flow

### 1. Create a Flox Environment (1-1.5 min)

```bash
mkdir my-demo-app && cd my-demo-app
flox init
flox install python nodejs
```

Show the manifest:
```bash
cat .flox/env/manifest.toml
```

```toml
[install]
python = {}
nodejs = {}
```

That's the entire environment definition. No version hunting, no compatibility tables, no manual steps.

### 2. Activate and Use (30-45 sec)

```bash
flox activate
python --version
node --version
which python
```

These come from the Flox environment, not the system. Completely isolated.

### 3. Share and Reproduce (1-1.5 min)

```bash
flox push
```

Now anyone can pull this exact environment:

```bash
flox activate -r yourname/my-demo-app
python --version   # Same version, same everything
```

Same manifest, same environment. No README to follow, no manual steps.

### 4. Key Differences (30 sec)

- **Unlike Docker**: Flox doesn't containerize. It's your development environment, running natively.
- **Unlike virtualenv/conda**: Flox handles everything, not just Python. Your whole stack.
- **Unlike package managers**: Flox is declarative and reproducible. Same manifest always gives the same environment.

### 5. Transition to GPU Demos (30 sec)

Those challenges — same environment everywhere, no manual version hunting — are **10x worse** with CUDA. The NVIDIA driver has to be on the host (it's a kernel module), but everything above the driver — the CUDA toolkit, compiler, runtime, libraries — that's what Flox manages. And you don't have to install the whole toolkit; you can install just the pieces you need.

## Q&A

**Q: Is this like Docker?**
A: Similar goal (reproducibility), different approach. Flox is for development environments, runs natively. Docker is for deployment and isolation via containers. They complement each other.

**Q: What about language-specific tools like virtualenv, npm, bundler?**
A: Flox includes those tools and more. It manages your entire environment — language runtime, system libraries, tools. Think of it as a layer above those tools.

**Q: Does this work on Windows?**
A: Flox works on macOS and Linux natively. Windows support is via WSL2.

**Q: How big are these environments?**
A: Packages are shared across environments via Nix store. Reuses common dependencies efficiently.

**Q: Can I use this in production?**
A: Yes. Many teams do. You can also use Flox to build containers for production deployment.

**Q: What package repositories does Flox use?**
A: Flox uses nixpkgs, which has 100,000+ packages. Most common tools and languages are available.

## Setup Requirements

- Flox installed
- Terminal
- Optionally: a second machine or VM to demonstrate portability

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Flox command not found | Verify installation, check PATH |
| Package not available | Search with `flox search <package>` |
| Activation seems slow first time | Expected — Flox is downloading packages. Subsequent activations are fast. |
| Environment doesn't work on second machine | Verify Flox is installed, check authentication |
