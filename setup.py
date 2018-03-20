from distutils.core import setup

PROJECT = 'td-gammon'
setup(
    name=PROJECT,
    packages=[
        'gammon',
        'gammon.agents',
        '',
    ],
    version='0.1.0',

    url='https://github.com/3cky/td-gammon',
)
