def test_column_transforms(sample_grizzlies_df):
    """Test column transformations."""
    # Add a new column
    sample_grizzlies_df['D'] = sample_grizzlies_df['A'] * 2
    assert sample_grizzlies_df['D'].tolist() == [2, 4, 6, 8, 10]
    
    # Apply a string operation
    sample_grizzlies_df['E'] = sample_grizzlies_df['B'].str.upper()
    assert sample_grizzlies_df['E'].tolist() == ['A', 'B', 'C', 'D', 'E']
    
    # Apply a numeric operation
    sample_grizzlies_df['F'] = sample_grizzlies_df['C'].round()
    assert sample_grizzlies_df['F'].tolist() == [1.0, 2.0, 3.0, 4.0, 6.0]