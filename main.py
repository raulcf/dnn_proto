#from keras.models import Sequential
#from keras.layers.core import Dense, Activation
#from keras.optimizers import SGD

import time
import numpy as np

from inputoutput import inputoutput as IO
from featurex import featurex as FX
from trainer import modelbuilder as trainer

def main():
    print()
#    model = Sequential()
#
#    model.add(Dense(output_dim=64, input_dim=100, init="glorot_uniform"))
#    model.add(Activation("relu"))
#    model.add(Dense(output_dim=10, init="glorot_uniform"))
#    model.add(Activation("softmax"))
#
#    model.compile(loss='categorical_crossentropy',
#                        optimizer=SGD(lr=0.01,
#                                      momentum=0.9, 
#                                      nesterov=True))

#    x = ["RRRPRPPYLPRPRPPPFFPPRLPPRIPPGFPPRFPPRFP",
#         "GNNRPVYIPQPRPPHPRI",
#         "TRSSRAGLQFPVGRVHRLLRK",
#         "ALWKTMLKKLGTMALHAGKAALGAAADTISQGTQ",
#         "GWGSFFKKAAHVGKHVGKAALTHYL"]

#    x = [
#        [2,3,4,5],
#        [9,9,9,9],
#        [9,8,9,9],
#        [1,2,3,4],
#        [1,2,4,2],
#    ]
#    y = [0,1,1,0,0]
#
#    st = time.time()
#    print(str(len(x)))
#    print(str(len(y)))
#    model.fit(x, y, nb_epoch=5, batch_size=32)
#    et = time.time()
#    print("Time to train: " + str(et-st))
#
#    objective_score = model.evaluate(x, y, batch_size=32)



#    x_test = ["RRRPRPPYLPPPRPFPFFPPRLPPRIPPGFPPRFPPRFP",
#              "GNNRPVYIPQPRPPHPRI",
#              "TRSSRAGLQFPVGRVVRLLRK",
#              "ALKWTMLKLKTGMALHAGKAALGAAADTISQGTQ",
#              "GWGSFFKKAAHVGKHVGKAALTHYL"]


#    x_test = [
#        [2,2,2,2],
#        [1,1,1,1],
#        [1,2,3,3],
#        [7,8,9,8],
#        [9,9,7,9]
#    ]
#    y_test = [0,1,1,0,0]
#    objective_score = model.evaluate(x_test, y_test, batch_size=32)
#    print("SCORE: " + str(objective_score))
#
#    classes = model.predict_classes(x_test, batch_size=32)
#    for x in classes:
#        print("CLASS: " + str(x))

def read_fasta_test():
    from Bio import SeqIO
    seqs = []
    for seq_record in SeqIO.parse("ls_orchid.fasta", "fasta"):
        print(seq_record.id)
        print(repr(seq_record.seq))
        print(len(seq_record))
        seqs.append(repr(seq_record.seq))
    return seqs

if __name__ == "__main__":
    #main()
    #seqs = IO.read_fasta_file("small_peptides.fasta")
    seqs = IO.read_fasta_file("ls_orchid.fasta")
    print(str(len(seqs)))
    #for s in seqs:
    #    print(str(s))
    seqs = seqs[:4]
    dvecs = []
    for s in seqs:
        if s is not None and s != "":
            dvec = FX.extract_named_descriptors_of_seq(s)
            dvecs.append(dvec)
    nvecs = []
    for v in dvecs:
        nv = FX.num_vector_from_descriptor_vector(v)
        nvecs.append(nv)
    M = trainer.build_sequential_model()
    x_b = [x for x in nvecs]
    x_batch = np.array(x_b)
    y_batch = [x%2 for x in range(1, len(x_batch)+1)]
    print("About to start training process with x.size: " + str(x_batch.shape))
    trained_M = trainer.fit_model_batch(M, x_batch, y_batch)
    classes = trainer.predict_with_model(x_batch, M)
    for c in classes:
        print(str(c))
    
    
    
    #vecs = [FX.num_vector_from_descriptor_vector(v) for v in dvecs]
    #for v in vecs:
    #    print(str(v))
