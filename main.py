#from keras.models import Sequential
#from keras.layers.core import Dense, Activation
#from keras.optimizers import SGD

import time
import numpy as np

from inputoutput import inputoutput as IO
from featurex import featurex as FX
from trainer import modelbuilder as trainer


def main():

    s_read_seq = time.time()
    seqs = IO.read_fasta_file("ls_orchid.fasta")
    e_read_seq = time.time()
    print("Total time to read sequences: " + str(e_read_seq - s_read_seq))
    print(str(len(seqs)))
    #for s in seqs:
    #    print(str(s))

    s_x_desc = time.time()
    seqs = seqs[:4]
    dvecs = []
    total_seqs = len(seqs)
    current_seq = 0
    for s in seqs:
        print("Extracting descriptors for sequence: " + str(current_seq) + "/" + str(total_seqs))
        s_x_seq = time.time()
        if s is not None and s != "":
            dvec = FX.extract_named_descriptors_of_seq(s)
            dvecs.append(dvec)
        e_x_seq = time.time()
        print("Took: " + str(e_x_seq - s_x_seq))
        print(" ")
        current_seq += 1
    e_x_desc = time.time()
    print("Total time to extract descriptors: " + str(e_x_desc - s_x_desc))

    IO.serialize_descriptor_vector(dvec)

    for k, v in dvec.items():
        print(str(k) + " -> " + str(v))

    """
    nvecs = []
    for v in dvecs:
        nv = FX.num_vector_from_descriptor_vector(v)
        nvecs.append(nv)
    M = trainer.build_sequential_model()
    x_b = [x for x in nvecs]
    x_batch = np.array(x_b)
    y_batch = [x % 2 for x in range(1, len(x_batch) + 1)]
    print("About to start training process with x.size: " + str(x_batch.shape))
    trained_M = trainer.fit_model_batch(M, x_batch, y_batch)
    classes = trainer.predict_with_model(x_batch, M)
    for c in classes:
        print(str(c))
    """


if __name__ == "__main__":
    main()

