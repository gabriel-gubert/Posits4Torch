from setuptools import setup

setup(
        name = 'Posits4Torch',
        version = '1.0.0',
        description = '',
        long_description = '',
        long_description_content_type = 'text',
        author = 'Gubert, G. V. K.',
        author_email = 'gvkg97@gmail.com',
        maintainer = 'Gubert, G. V. K.',
        maintainer_email = 'gvkg97@gmail.com',
        url = '',
        download_url = '',
        packages = ['Posits4Torch'],
        package_dir = {'': 'src'},
        license = 'Apache License 2.0',
        zip_safe = True,
        install_requires = [
            'numpy',
            'torch'
        ]
)
