def parser(cfg):

    blocks = []
    with open(cfg) as fi:
        lines = fi.read().split('\n')
    #Remove left Whitespaces 
    lines = [l.strip() for l in lines]
    #Remove comments
    lines = [l for l in lines if len(l) > 0 and l[0] != '#']
    block = {}
    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].strip()
        else:
            key, value = line.split('=')
            key = key.strip()
            value = value.strip()
            block[key] = value
    blocks.append(block)

    return blocks