'''
    Responsible for creating ML application as a package and deloy it
'''

from setuptools import find_packages, setup
from typing import List


HYPHEN_E_DOT = '-e'

def get_requirements(file_path:str)->List[str]:
    '''
        this function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        reruirements=file_obj.readlines()
        reruirements=[req.replace('\n', '') for req in reruirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)


setup(
    name='ML Project',
    version='0.0.1',
    author='Usama',
    author_email='usamafzal36@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)