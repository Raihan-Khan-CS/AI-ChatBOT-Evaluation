from setuptools import setup, find_packages

setup(
    name="ai-chatbot-evaluation",
    version="1.0.0",
    description="Evaluating Accuracy, Authenticity, and Up-to-dateness in AI Chatbots",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="MD Raihan Khan, Zannatul Islam Proma",
    author_email="raihan.khan.242@northsouth.edu",
    url="https://github.com/Raihan-Khan-CS/AI-ChatBOT-Evaluation",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "spacy>=3.7.0",
        "nltk>=3.8.1",
        "sentence-transformers>=2.2.2",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "numpy>=1.24.0",
        "pandas>=2.1.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
