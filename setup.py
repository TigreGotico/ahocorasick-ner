import os.path

import os
from setuptools import setup

BASEDIR = os.path.abspath(os.path.dirname(__file__))


def get_version():
    """ Find the version"""
    version_file = os.path.join(BASEDIR, 'ahocorasick_ner', 'version.py')
    major, minor, build, alpha = (None, None, None, None)
    with open(version_file) as f:
        for line in f:
            if 'VERSION_MAJOR' in line:
                major = line.split('=')[1].strip()
            elif 'VERSION_MINOR' in line:
                minor = line.split('=')[1].strip()
            elif 'VERSION_BUILD' in line:
                build = line.split('=')[1].strip()
            elif 'VERSION_ALPHA' in line:
                alpha = line.split('=')[1].strip()

            if ((major and minor and build and alpha) or
                    '# END_VERSION_BLOCK' in line):
                break
    version = f"{major}.{minor}.{build}"
    if int(alpha):
        version += f"a{alpha}"
    return version


def required(requirements_file):
    """ Read requirements file and remove comments and empty lines. """
    with open(os.path.join(BASEDIR, requirements_file), 'r') as f:
        requirements = f.read().splitlines()
        return [pkg for pkg in requirements
                if pkg.strip() and not pkg.startswith("#")]

PLUGIN_ENTRY_POINT = 'ovos-ahocorasick-ner-plugin=ahocorasick_ner.opm:AhocorasickNERTransformer'


setup(
    name="ahocorasick-ner",
    version=get_version(),
    modules=["ahocorasick_ner"],
    install_requires=required("requirements.txt"),
    entry_points={'opm.transformer.intent': PLUGIN_ENTRY_POINT},
    author="JarbasAI",
    author_email="jarbasai@mailfence.com",
    description="A fast, dictionary-based Named Entity Recognition system using the Aho-Corasick algorithm.",
    # long_description=open("README.md").read(),
    # long_description_content_type="text/markdown",
    url="https://github.com/TigreGotico/ahocorasick-ner",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Text Processing :: Linguistic",
        "Intended Audience :: Developers",
    ],
    python_requires=">=3.9",
)
