from setuptools import setup, find_packages

setup(
    name="karanir",
    version="1.0",
    author="Yiqi Sun (Zongjing Li)",
    author_email="ysun697@gatech.edu",
    description="Karanir, the personal package for machine learning and more",

    # project main page
    url="http://jiayuanm.com/", 

    # the package that are prerequisites
    packages=find_packages(),
    include_package_data = True,
    package_data={
        },
    
)

"""
'':['moic',
        'moic/mklearn',
        'moic/learn/nn'],
        'moic': ['mklearn'],
        'bandwidth_reporter':['moic','moic/mklearn','moic/learn/nn']
               
"""