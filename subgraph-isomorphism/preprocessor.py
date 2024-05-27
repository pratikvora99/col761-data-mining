import sys
import pickle

filepath = sys.argv[1]
algo = sys.argv[2]

newLines = []
vmap = {}
vcount = 0
if(len(sys.argv) > 3):
    pickle_path = sys.argv[3]
    with open(pickle_path, "rb") as fp:
        vmap = pickle.load(fp)
    vcount = max([int(val) for val in vmap.values()])
tmap = {}
tcount = 0
filename = filepath.split(".")[0]
with open(filepath,'r') as fp:
    nv = 0
    nvflag = False
    neflag = False
    vi = 0
    ne = 0
    ei = 0
    for line in fp:
        if(ei < ne):
            if(algo == "gspan" or algo == "gaston"):
                newLines.append("e " + line.strip())
            elif(algo == "fsg"):
                newLines.append("u " + line.strip())
            ei+=1
        if(neflag):
            ne = int(line.strip())
            neflag = False
            ei=0
        if(vi < nv):
            if(algo == "gaston" or algo=="gspan"):
                dummy = line.strip()
                if(dummy not in vmap.keys()):
                    vmap[dummy] = str(vcount)
                    vcount+=1
                newLines.append("v " + str(vi) + " " + vmap[dummy])
            else:
                newLines.append("v " + str(vi) + " " + line.strip())
            vi+=1
            if(vi == nv):
                neflag = True
        if(nvflag):
            nv = int(line.strip())
            nvflag = False
            vi=0
        if(line.__contains__('#')):
            dummy = line.strip().split("#")[1]
            tmap[dummy] = tcount
            newLines.append("t # "+dummy)
            tcount += 1
            nvflag = True
with open('formatted_file', 'w+') as fp:
    fp.writelines([f'{x}\n' for x in newLines])
with open('vmap.pkl', 'wb') as fp:
    pickle.dump(vmap, fp)
with open('tmap.pkl', 'wb') as fp:
    pickle.dump(tmap, fp)
