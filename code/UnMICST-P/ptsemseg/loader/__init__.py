import json

# Training dataloaders
from ptsemseg.loader.DNA_Aug import DNA_Aug
from ptsemseg.loader.DNA_GaussianAug import DNA_GaussianAug
from ptsemseg.loader.DNA_NES_Aug import DNA_NES_Aug
from ptsemseg.loader.DNA_NES_NoAug import DNA_NES_NoAug
from ptsemseg.loader.DNA_NoAug import DNA_NoAug
from ptsemseg.loader.NES import NES

# Testing dataloaders
from ptsemseg.loader.DNA_test import DNA_test
from ptsemseg.loader.DNA_NES_test import DNA_NES_test
from ptsemseg.loader.NES_test import NES_test

def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        "DNA_Aug": DNA_Aug,
        "DNA_GaussianAug": DNA_GaussianAug,
        "DNA_NES_Aug": DNA_NES_Aug,
        "DNA_NES_NoAug": DNA_NES_NoAug,
        "DNA_NoAug": DNA_NoAug,
        "NES": NES,
        "DNA_test": DNA_test,
        "DNA_NES_test": DNA_NES_test,
        "NES_test": NES_test,

    }[name]
