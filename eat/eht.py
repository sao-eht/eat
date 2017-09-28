import csv
import pkg_resources as pr

class eht:
    """
    A class describing the EHT telescope array
    """
    def __init__(self):
        modpath  = __name__.split(".")[0]
        fullname = pr.resource_filename(modpath, "data/sites.csv")
        with open(fullname, "r") as handle:
            reader = csv.DictReader(handle)
            self.array = {dict['site']: dict for dict in reader}
