import math

def convert_to_xyz(latitude, longitude, altitude):
    # Convert latitude and longitude to radians
    lat_rad = math.radians(latitude)
    lon_rad = math.radians(longitude)

    # Earth radius in meters
    earth_radius = 6378.137

    # Calculate XYZ coordinates
    x = (earth_radius + altitude) * math.cos(lat_rad) * math.cos(lon_rad)
    y = (earth_radius + altitude) * math.cos(lat_rad) * math.sin(lon_rad)
    z = (earth_radius + altitude) * math.sin(lat_rad)

    return x, y, z

# Example usage
latitude = 37.7749
longitude = -122.4194
altitude = 0

x, y, z = convert_to_xyz(latitude, longitude, altitude)
print(f"X: {x}, Y: {y}, Z: {z}")