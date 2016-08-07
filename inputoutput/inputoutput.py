import pickle
import config as C

from Bio import SeqIO


def read_fasta_file(filename):
    '''
    Returns list of sequence records in a fasta file
    '''
    seqs = []
    for sr in SeqIO.parse(filename, "fasta"):
        srecord = sr.seq
        seqs.append(srecord)
    return seqs


def serialize_descriptor_vector(dvec):
    path = C.serde_model_path + C.model_name + ".pickle"
    output = open(path, 'wb')
    pickle.dump(dvec, output)
    output.close()


def deserialize_descriptor_vector():
    path = C.serde_model_path + C.model_name + ".pickle"
    input = open(path, 'rb')
    dvec = pickle.load(input)
    return dvec

    
if __name__ == "__main__":
    filename = "ls_orchid.fasta"
    seqs = read_fasta_file(filename)
    for s in seqs:
        print(str(s))
