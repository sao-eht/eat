import csv
import pkg_resources as pr

class eht:
    """
    A class describing the EHT telescope array
    """
    def __init__(self, sites):
        modpath  = __name__.split(".")[0]
        fullname = pr.resource_filename(modpath, "data/sites.csv")
        with open(fullname, "r") as handle:
            reader = csv.DictReader(handle)
            if sites is None:
                self.array = {d['site']: d
                              for d in reader}
            elif isinstance(sites, str):
                self.array = {d['site']: d
                              for d in reader
                              if  d['site_id'] is not ''
                              and d['site_id'] in sites}
            else:
                self.array = {d['site']: d
                              for d in reader
                              if  d['site'] in sites}
