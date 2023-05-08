labels = [{'name':'a', 'id':1}, {'name':'b', 'id':2}, {'name':'c', 'id':3},
          {'name':'d', 'id':4}, {'name':'e', 'id':5}, {'name':'f', 'id':6}, 
          {'name':'g', 'id':7}]

with open('LABEL_MAP', 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')