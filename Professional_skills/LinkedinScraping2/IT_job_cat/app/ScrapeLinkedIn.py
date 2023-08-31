# import requests

# def scrape_linkedin_skills(linkedin_profile_url):
#     api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
#     api_key = 'jtiOFtUah7GkK79fcRQqww'
#     headers = {'Authorization': 'Bearer ' + api_key}
   

#     response = requests.get(api_endpoint,
#                             params={'url': linkedin_profile_url, 'skills': 'include'},
#                             headers=headers)

#     profile_data = response.json()

#     if 'skills' in profile_data:
#         return profile_data['skills']
#     else:
#         return []

# # Your API key
# api_key = 'jtiOFtUah7GkK79fcRQqww'
# # Example LinkedIn profile URL
# linkedin_profile_url = 'https://www.linkedin.com/in/chanuxbro/'

# # Call the function to get skills
# skills = scrape_linkedin_skills(linkedin_profile_url)
# print("Skills:", skills)


import requests

def scrape_linkedin_skills(linkedin_profile_url):
    api_key = 'MxVwlMuCI00hrmsugxWLjA'
    api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
    headers = {'Authorization': 'Bearer ' + api_key}

    response = requests.get(api_endpoint,
                            params={'url': linkedin_profile_url, 'skills': 'include'},
                            headers=headers)

    profile_data = response.json()

    if 'skills' in profile_data:
        return profile_data['skills']
    else:
        return []

# # Example LinkedIn profile URL
# linkedin_profile_url = 'https://www.linkedin.com/in/chanuxbro/'

# # Call the function to get skills
# skills = scrape_linkedin_skills(linkedin_profile_url)
# print("Skills:", skills)
