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
    
if __name__ == "__main__":
    filename = "ls_orchid.fasta"
    seqs = read_fasta_file(filename)
    for s in seqs:
        print(str(s))
