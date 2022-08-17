# this file contains some functions for manipulation of the data

def encode_family_to_number(family):
    """
    Takes the family (e.g. Books) and returns the corresponding number. For this function to work it is assumed that
    the families are labeled with numbers ranging from 0,..,32. s.t. AUTOMOTIVE=0,...,SEAFOOD=32
    :param family: str, name of the family
    :return: int, corresponding number encoding of the family
    """
    family_lst = ['AUTOMOTIVE','BABY CARE','BEAUTY','BEVERAGES','BOOKS','BREAD/BAKERY','CELEBRATION',
                 'CLEANING','DAIRY','DELI', 'EGGS', 'FROZEN FOODS','GROCERY I','GROCERY II', 'HARDWARE',
                 'HOME AND KITCHEN I','HOME AND KITCHEN II','HOME APPLIANCES',  'HOME CARE', 'LADIESWEAR',
                 'LAWN AND GARDEN','LINGERIE','LIQUOR,WINE,BEER','MAGAZINES','MEATS', 'PERSONAL CARE',
                 'PET SUPPLIES','PLAYERS AND ELECTRONICS','POULTRY', 'PREPARED FOODS','PRODUCE',
                 'SCHOOL AND OFFICE SUPPLIES','SEAFOOD']
    return family_lst.index(family)

def decode_family_from_number(number):
    """
    Takes a number between 0,...,32 and returns the corresponding family
    :param number: int, number of the family
    :return: str, family name
    """
    if not isinstance(number,int) or  number > 32 or number < 0:
        raise ValueError('Input must be an integer between 0 and 32')

    family_lst = ['AUTOMOTIVE','BABY CARE','BEAUTY','BEVERAGES','BOOKS','BREAD/BAKERY','CELEBRATION',
                 'CLEANING','DAIRY','DELI', 'EGGS', 'FROZEN FOODS','GROCERY I','GROCERY II', 'HARDWARE',
                 'HOME AND KITCHEN I','HOME AND KITCHEN II','HOME APPLIANCES',  'HOME CARE', 'LADIESWEAR',
                 'LAWN AND GARDEN','LINGERIE','LIQUOR,WINE,BEER','MAGAZINES','MEATS', 'PERSONAL CARE',
                 'PET SUPPLIES','PLAYERS AND ELECTRONICS','POULTRY', 'PREPARED FOODS','PRODUCE',
                 'SCHOOL AND OFFICE SUPPLIES','SEAFOOD']
    return family_lst[number]
# test change for git