import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


class User:
    def __init__(self, name, gender, age, body_measurements, style_preferences):
        self.name = name
        self.gender = gender
        self.age = age
        self.body_measurements = body_measurements
        self.style_preferences = style_preferences


class Outfit:
    def __init__(self, name, style, color, brand, image):
        self.name = name
        self.style = style
        self.color = color
        self.brand = brand
        self.image = image


class VirtualStylist:
    def __init__(self):
        self.users = []
        self.outfits = []
        self.user_profiles = None
        self.outfit_data = None

    def create_user_profile(self, name, gender, age, body_measurements, style_preferences):
        user = User(name, gender, age, body_measurements, style_preferences)
        self.users.append(user)

    def add_outfit(self, name, style, color, brand, image):
        outfit = Outfit(name, style, color, brand, image)
        self.outfits.append(outfit)

    def generate_user_profiles(self):
        profile_names = []
        body_measurements = []
        style_preferences = []
        for user in self.users:
            profile_names.append(user.name)
            body_measurements.append(' '.join(user.body_measurements))
            style_preferences.append(' '.join(user.style_preferences))

        self.user_profiles = pd.DataFrame({
            'Name': profile_names,
            'Body Measurements': body_measurements,
            'Style Preferences': style_preferences
        })

    def generate_outfit_data(self):
        outfit_names = []
        styles = []
        colors = []
        brands = []
        for outfit in self.outfits:
            outfit_names.append(outfit.name)
            styles.append(outfit.style)
            colors.append(outfit.color)
            brands.append(outfit.brand)

        self.outfit_data = pd.DataFrame({
            'Outfit': outfit_names,
            'Style': styles,
            'Color': colors,
            'Brand': brands
        })

    def recommend_outfit(self, user_name):
        user_index = self.user_profiles[self.user_profiles['Name']
                                        == user_name].index[0]

        count_vectorizer = CountVectorizer()
        user_vector = count_vectorizer.fit_transform(
            [self.user_profiles.loc[user_index, 'Body Measurements'] + ' ' + self.user_profiles.loc[user_index, 'Style Preferences']])
        outfit_vectors = count_vectorizer.transform(
            self.outfit_data['Style'] + ' ' + self.outfit_data['Color'] + ' ' + self.outfit_data['Brand'])
        similarity_scores = cosine_similarity(user_vector, outfit_vectors)[0]

        recommended_outfits = self.outfit_data.iloc[np.argsort(
            -similarity_scores)][:3]

        return recommended_outfits

    def display_recommended_outfits(self, user_name):
        recommended_outfits = self.recommend_outfit(user_name)
        print(f"Recommended Outfits for {user_name}:")
        print(recommended_outfits)


def main():
    stylist = VirtualStylist()

    stylist.create_user_profile('Alice', 'Female', 25, [
                                '32-26-36'], ['Casual', 'Boho'])
    stylist.create_user_profile(
        'Bob', 'Male', 30, ['40-32-38'], ['Formal', 'Classic'])
    stylist.create_user_profile('Claire', 'Female', 22, [
                                '34-28-38'], ['Girly', 'Vintage'])

    stylist.add_outfit('Outfit 1', 'Boho', 'Blue', 'Zara', 'outfit1.jpg')
    stylist.add_outfit('Outfit 2', 'Classic', 'Black', 'Gucci', 'outfit2.jpg')
    stylist.add_outfit('Outfit 3', 'Girly', 'Pink',
                       'Ralph Lauren', 'outfit3.jpg')

    stylist.generate_user_profiles()
    stylist.generate_outfit_data()

    stylist.display_recommended_outfits('Alice')


if __name__ == "__main__":
    main()
