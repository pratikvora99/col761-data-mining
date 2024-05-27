import sys
import pickle

with open('vmap.pkl', "rb") as fp:
    vmap = pickle.load(fp)
inv_vmap = {v: k for k, v in vmap.items()}

with open('tmap.pkl', "rb") as fp:
    tmap = pickle.load(fp)
inv_tmap = {v: k for k, v in tmap.items()}

with open('index_raw.fp', 'w+') as f:
    with open(sys.argv[1],'r') as fp:
        for line in fp:
            line = line.strip()
            if(line!="" and line[0]=='v'):
                line = line.split()
                line[2] = inv_vmap[line[2]]
                line = ' '.join(line)
            if(line!="" and line[0]=='x'):
                line = line.split()
                for i in range(1,len(line)):
                    line[i] = str(inv_tmap[int(line[i])])
                line = ' '.join(line)
            f.writelines([line+'\n'])
            
# is_root = {}
# heirarchyMap = {}
# with open("formatted_file.pc", "r") as fp:
#     for line in fp:
#         sid_list = line.strip().split()
#         sid = sid_list[0]
#         is_root[sid] = True
#         children = []
#         for i in range(1, len(sid_list)):
#             children.append(sid_list[i])
#             if(sid_list[i] in heirarchyMap):
#                 is_root[sid_list[i]] = False
#                 children += heirarchyMap[sid_list[i]]
#         heirarchyMap[sid] = children

# with open("heirarchy.txt", "w+") as fp:
#     for sid in heirarchyMap:
#         # print("yes")
#         if(is_root[sid] or True):
#             fp.write(sid)
#             for ssid in heirarchyMap[sid]:
#                 fp.write(" {}".format(ssid))
#             fp.write("\n")

# c2pHeirarchyMap = {}
# with open("formatted_file.pc", "r") as fp:
#     for line in fp:
#         sid_list = line.strip().split()
#         parent = sid_list[0]
#         c2pHeirarchyMap[parent] = []
#         for child in sid_list[1:]:
#             c2pHeirarchyMap[child].append(parent)

# with open("c2pheirarchy.txt", "w+") as fp:
#     for sid in c2pHeirarchyMap:
#         fp.write(sid)
#         for ssid in c2pHeirarchyMap[sid]:
#             fp.write(" {}".format(ssid))
#         fp.write("\n")

# linesToBeRead = []
# with open("formatted_file.fp", "r+") as fp:
#     linesToBeRead = [line.strip() for line in fp]
# linesToWrite = []
# first = True
# for line in linesToBeRead:
#     if(line=='' or line[0]=='#'):
#         continue
#     if(line[0]=='t'):
#         if(first):
#             first = False
#         else:
#             linesToWrite.append("\n")
#     if(line[0]=='u'):
#         line = 'v' + line[1:]
#     linesToWrite.append(line)
# with open("formatted_file2.fp", "w+") as fp:
#     fp.writelines([f'{x}\n' for x in linesToWrite])