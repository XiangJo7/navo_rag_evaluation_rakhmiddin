from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

def create_metadata(df):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    return metadata

def generate_synthetic_data(df, n_samples):
    metadata = create_metadata(df)

    synthesizer = CTGANSynthesizer(metadata)
    synthesizer.fit(df)

    synthetic_df = synthesizer.sample(n_samples)
    return synthetic_df
