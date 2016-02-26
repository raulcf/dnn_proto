from pydpi.pydrug import Chem
from pydpi.drug import constitution


def extract_named_descriptors_of_seq(sequence):
    mol = Chem.MolFromSequence(str(sequence))
    res = None
    if mol is not None:
        res = constitution.GetConstitutional(mol)
    return res

def num_vector_from_descriptor_vector(descriptor_vector):
    x = []
    for k, v in descriptor_vector.items():
        x.append(v)
    return x

if __name__ == "__main__":
    print("TODO")
