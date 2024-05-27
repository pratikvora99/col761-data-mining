import sys
newLines = []
filepath = sys.argv[1]
algo = sys.argv[2]
vmap = {}
vcount = 0
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
            newLines.append("t # "+str(tcount))
            tcount += 1
            nvflag = True
with open(filename+'_'+algo, 'w+') as fp:
    fp.writelines([f'{x}\n' for x in newLines])
