import json

def network_to_json( network ):
    filename = 'acc.{}_opt.{}_act.{}.txt'.format(network.accuracy, 
        network.params['optimizer'],
        network.params['ann_activation'] )
    fp = open( filename, mode='w' )
    json.dump( network.params, fp, indent=4 )

# test
# network_to_json(None)
