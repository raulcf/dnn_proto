#from keras.models import Sequential
#from keras.layers.core import Dense, Activation
#from keras.optimizers import SGD

import time
import numpy as np
from random import shuffle

from inputoutput import inputoutput as IO
from featurex import featurex as FX
from trainer import modelbuilder as trainer


def main(model=None):

    """
    extract_descriptors_from_fasta_to_file("neg_peptides_sample.fasta", "neg_p")
    extract_descriptors_from_fasta_to_file("pos_peptides.fasta", "pos_p")
    """

    print("Deserializing descriptor vectors...")
    neg_dvec = IO.deserialize_descriptor_vector("neg_p")
    pos_dvec = IO.deserialize_descriptor_vector("pos_p")
    print("Deserializing descriptor vectors...OK")
    print("")

    print("Extracting numerical vectors...")

    neg_nmat = []
    for dvec in neg_dvec:
        if dvec is None:
            continue
        neg_nvec = FX.num_vector_from_descriptor_vector(dvec)
        neg_nmat.append(neg_nvec)

    pos_nmat = []
    for dvec in pos_dvec:
        if dvec is None:
            continue
        pos_nvec = FX.num_vector_from_descriptor_vector(dvec)
        pos_nmat.append(pos_nvec)

    print("Extracting numerical vectors...OK")
    print("")

    print("Preparing training and label data...")

    # Prepare labels
    neg_y_batch = [0 for _ in neg_nmat]
    pos_y_batch = [1 for _ in pos_nmat]

    # Append training data and labels, and create shuffle idx
    neg_nmat.extend(pos_nmat)
    x_batch_appended = neg_nmat
    neg_y_batch.extend(pos_y_batch)
    y_batch_appended = neg_y_batch

    x_batch_appended = np.array(x_batch_appended)
    v_size = len(y_batch_appended)

    y_batch_appended = np.array(y_batch_appended)

    print("Creating shuffle idx array, size: " + str(v_size))

    shuffle_idx_array = range(0, v_size, 1)
    shuffle(shuffle_idx_array)

    # Finally shuffle to get the real x and y training and labels
    x_batch = x_batch_appended[shuffle_idx_array]
    y_batch = y_batch_appended[shuffle_idx_array]
    print("Preparing training and label data...OK")
    print("")

    if model is None:
        print("Training model on data...")
        s_training = time.time()
        M = trainer.build_sequential_model()
        trained_M = trainer.fit_model_batch(M, x_batch, y_batch, num_epoch=100)
        e_training = time.time()
        print("Training model on data...OK, took: " + str((e_training - s_training)))

        # Store expensive-to-train model
        IO.serialize_model(trained_M, "models/basic_sequential")
    else:
        trained_M = model

    print("Classifying data...")
    s_classify = time.time()
    classes = trainer.predict_with_model(x_batch, trained_M)
    e_classify = time.time()
    print("Classifying data...OK, took: " + str((e_classify - s_classify)))

    total_samples = 0
    hits = 0
    for i in range(len(classes)):
        total_samples = total_samples + 1
        #print("Ground truth: " + str(y_batch[i]))
        #print("Prediction: " + str(classes[i]))
        if y_batch[i] == classes[i]:
            hits = hits + 1

    print("Hits/Total: " + str(hits) + "/" + str(total_samples))

    """
    s_read_seq = time.time()
    seqs = IO.read_fasta_file("small_peptides.fasta")
    e_read_seq = time.time()
    print("Total time to read sequences: " + str(e_read_seq - s_read_seq))
    print(str(len(seqs)))
    #for s in seqs:
    #    print(str(s))

    s_x_desc = time.time()
    #seqs = seqs[:4]
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

    IO.serialize_descriptor_vector(dvecs)

    for dv in dvecs:
        for k, v in dv.items():
            print(str(k) + " -> " + str(v))
    """

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


def extract_descriptors_from_fasta_to_file(fastafile, outputfile):
    print("Working on: " + str(fastafile))
    print(" ")
    s_read_seq = time.time()
    seqs = IO.read_fasta_file(fastafile)
    e_read_seq = time.time()
    print("Total time to read sequences: " + str(e_read_seq - s_read_seq))
    print(str(len(seqs)))

    s_x_desc = time.time()
    # seqs = seqs[:4]
    dvecs = []
    total_seqs = len(seqs)
    current_seq = 1
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

    IO.serialize_descriptor_vector(dvecs, o_file=outputfile)


if __name__ == "__main__":

    #model = IO.deserialize_model("models/basic_sequential")
    model = None

    main(model)

