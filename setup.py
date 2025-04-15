from setuptools import setup

setup(
    name="ahocorasick-ner",
    version="0.0.1",
    py_modules=["ahocorasick_ner"],
    install_requires=[
        "pyahocorasick"
    ],
    author="JarbasAI",
    author_email="jarbasai@mailfence.com",
    description="A fast, dictionary-based Named Entity Recognition system using the Aho-Corasick algorithm.",
    #long_description=open("README.md").read(),
    #long_description_content_type="text/markdown",
    url="https://github.com/TigreGotico/ahocorasick-ner",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Text Processing :: Linguistic",
        "Intended Audience :: Developers",
    ],
    python_requires=">=3.7",
)
