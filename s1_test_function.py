def get_height_for_16_9(width: float) -> float:
    """
    Get the height value for a given width using 16:9 aspect ratio.
    
    Args:
        width (float): The width value
        
    Returns:
        float: The calculated height value
    """
    return (width * 9) / 16


if __name__ == "__main__":
    height = get_height_for_16_9(1920)
    print(height)