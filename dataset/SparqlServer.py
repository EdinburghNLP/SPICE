from pymantic import sparql

class SparqlServer(object):
    _instance = None

    def __init__(self):
        raise RuntimeError('Call instance() instead')

    @classmethod
    def instance(cls, renew=False):
        if cls._instance is None or renew:
            if renew:
                cls._instance.s.close()
            print('Creating new instance')
            cls._instance = sparql.SPARQLServer('http://localhost:9999/blazegraph/namespace/wd/sparql')
        return cls._instance
