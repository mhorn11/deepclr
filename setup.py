from setuptools import setup


# read requirements.txt
def _fix_requirement(req: str) -> str:
    if req.startswith('git'):
        return req.split('=')[-1]
    else:
        return req


with open('requirements.txt') as f:
    required = [_fix_requirement(req) for req in f.read().splitlines()]


# run setup (most arguments are defined in setup.cfg)
setup(
    install_requires=required
)
