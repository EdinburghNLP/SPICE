

class SparqlResults:

    @staticmethod
    def getEntitySetFromBindings(results):
        """
        :param results: {'head': {'vars': ['x']}, 'results':
                {'bindings': [{'x': {'type': 'uri', 'value': 'http://www.wikidata.org/entity/Q155'}},
                {'x': {'type': 'uri', 'value': 'http://www.wikidata.org/entity/Q159'}},
                {'x': {'type': 'uri', 'value': 'http://www.wikidata.org/entity/Q183'}}]}}
        :return: {'x': [Q155, Q159, Q183]}

        :param results: {'head': {}, 'boolean': False}
        :return: {'boolean': False}
        """
        if 'boolean' in results.keys():
            return {'boolean': results['boolean']}
        else:
            varBindings = {}
            for var in results['head']['vars']: # we expect to find one variable...!?
                varBindings[var] = []
                for bin in results['results']['bindings']:
                    if var in bin.keys():
                        varBindings[var].append(bin[var]['value'].split('/')[-1])

            return varBindings

