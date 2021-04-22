import setuptools

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f.readlines()]

setup_requirements = ["setuptools"]

setuptools.setup(name='vidaug',
                 version='0.1',
                 description='Video Augmentation Library',
                 url='https://github.com/okankop/vidaug',
                 author='Okan Kopuklu',
                 author_email='okankopuklu@gmail.com',
                 setup_requires=setup_requirements,
                 install_requires=requirements,
                 license='MIT',
                 packages=setuptools.find_packages(),
                 zip_safe=False)
