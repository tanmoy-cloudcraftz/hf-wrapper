from setuptools import setup, find_packages

REPO_NAME = "hf-wrapper"
AUTHOR_NAME = "tanmoy-cloudcraftz" 
AUTHOR_EMAIL = "tanmoy.roy@cloudcraftz.io"


try: 
    REQUIRES = list() 
    f = open("requirements.txt", "rb") 
    for line in f.read().decode("utf-8").split("\n"): 
        line = line.strip() 
        if "#" in line: 
            line = line[: line.find("#")].strip() 
        if line: 
            REQUIRES.append(line)
            
except FileNotFoundError: 
    print("'requirements.txt' not found!") 
    REQUIRES = list()


setup(
    name = REPO_NAME,
    version = "0.0.1",
    author = AUTHOR_NAME,
    author_email = AUTHOR_EMAIL,
    description = "Wrapper for huggingface library",
    url = f"https://github.com/{AUTHOR_NAME}/{REPO_NAME}",
    project_urls = {
        "Bug Tracker": f"https://github.com/{AUTHOR_NAME}/{REPO_NAME}/issues"
    },
    packages = find_packages(),
    install_requires = REQUIRES,
)

