import setuptools

setuptools.setup(
    name="package",
    version="0.0.1",
    author="DigiNova",
    author_email='info@diginova.com.tr',
    description="Package",
    url='https://github.com/novavision-ai/package',
    license='MIT',
    install_requires=['sdk', 'opencv-python-headless'],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    packages=[
        'novavision.package',
        'novavision.package.classes',
        'novavision.package.configs',
        'novavision.package.dataloaders',
        'novavision.package.executors',
        'novavision.package.models',
        'novavision.package.utils',
        'novavision.package.weights'
    ],
    package_dir={'novavision.package': 'src'},
    python_requires=">=3.6"
)