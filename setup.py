import os


def MakeDeepLearningStructure(foldername):

    #make the folder structure

    #make a the foldername
    os.makedirs(foldername, exist_ok=True)


    #make a folder name called data
    os.makedirs(foldername + '/data', exist_ok=True)

    #make __init__.py inside data
    os.system('touch ' + foldername + '/data/__init__.py')


    #make a folder name called models
    os.makedirs(foldername + '/models', exist_ok=True)

    #make __init__.py inside models
    os.system('touch ' + foldername + '/models/__init__.py')


    #make a folder name called utils
    os.makedirs(foldername + '/utils', exist_ok=True)

    #make __init__.py inside utils
    os.system('touch ' + foldername + '/utils/__init__.py')


    #make a folder called losses
    os.makedirs(foldername + '/losses', exist_ok=True)

    #make __init__.py inside losses
    os.system('touch ' + foldername + '/losses/__init__.py')


    #make a train.py
    os.system('touch ' + foldername + '/train.py')

    #make a test.py
    os.system('touch ' + foldername + '/test.py')

    #make a README.md
    os.system('touch ' + foldername + '/README.md')


foldername = '.'
MakeDeepLearningStructure(foldername)