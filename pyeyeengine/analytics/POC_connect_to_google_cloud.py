from bigquery import get_client

json_key = 'analytics-fad3d92a636a.json'

client = get_client(json_key_file=json_key, readonly=False)

rows =  [
    {'game_type': 'hi', 'game_name': '1'} # duplicate entry
]

inserted = client.push_rows('analytics', 'game_leave_join', rows, 'game_type')