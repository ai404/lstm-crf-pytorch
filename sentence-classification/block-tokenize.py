import sys

if __name__ == "__main__": # tokenize documents into blocks
    if len(sys.argv) != 3:
        sys.exit("Usage: %s sizes data" % sys.argv[0])
    fi = open(sys.argv[2])
    fo = open(sys.argv[2] + ".block", "w")
    data = fi.read().strip().split("\n\n")
    sizes = list(map(int, sys.argv[1].split(",")))
    blocks = dict()
    for doc in data:
        doc = doc.split("\n")
        for i in range(len(doc)):
            for z in sizes:
                blocks["\n".join(doc[i:i + z])] = True
    fo.write("\n\n".join(blocks.keys()) + "\n")
    fi.close()
    fo.close()
