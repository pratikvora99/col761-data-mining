import sys
import os
import time

dataset = sys.argv[1]
filename = 'formatted_file'

support = 0.40

# command = "python3 preprocessor.py " + dataset + " fsg"
command = "python3 preprocessor.py " + dataset + " gspan"
os.system(command)
command = "./gSpan-64 -s {} -o -i -m 1 -f {}".format(support, filename)
# command = "./fsg {} -s {} -m 1 -t -p".format(filename, support)
start = time.time()
os.system(command)
end = time.time()
print("Time taken to complete gspan with support: {} : {:.2f} ms".format(support, end - start))

output_file_name = filename+'.fp'
command = "python3 postprocessor.py {}".format(output_file_name)
os.system(command)

with open("index_stats.txt", "w+") as fp:
    fp.write(dataset+'\n')
    with open("index_raw.fp".format(filename), "r") as fp2:
        text = fp2.read()
        fp.write(str(text.count("t #")))