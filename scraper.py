import os
import time
import pandas as pd
import os

def scrape_historical_data(start_year, end_year, skaters, verbose):
    for year in range(start_year, end_year+1):
        if skaters == True:
            filename = f'{year-1}-{year}_skater_data.csv'
            file_path = os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Historical Skater Data', filename)
        else:
            filename = f'{year-1}-{year}_goalie_data.csv'
            file_path = os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Historical Goaltending Data', filename)

        if os.path.exists(file_path):
            if verbose:
                print(f'{filename} already exists in the following directory: {file_path}')
            continue

        if skaters == True:
            url = f"https://www.naturalstattrick.com/playerteams.php?fromseason={year-1}{year}&thruseason={year-1}{year}&stype=2&sit=all&score=all&stdoi=std&rate=n&team=ALL&pos=S&loc=B&toi=0&gpfilt=none&fd=&td=&tgp=410&lines=single&draftteam=ALL"
        else:
            url = f"https://www.naturalstattrick.com/playerteams.php?fromseason={year-1}{year}&thruseason={year-1}{year}&stype=2&sit=all&score=all&stdoi=g&rate=n&team=ALL&pos=S&loc=B&toi=0&gpfilt=none&fd=&td=&tgp=410&lines=single&draftteam=ALL"
        df = pd.read_html(url)[0]
        df = df.iloc[:, 1:]
        if verbose == True:
            print(df)

        export_path = os.path.dirname(file_path)
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        df.to_csv(os.path.join(export_path, filename))
        if verbose:
            print(f'{filename} has been downloaded to the following directory: {export_path}')

def aggregate_training_data():
    file_path = os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Historical Skater Data')
    files = sorted(os.listdir(file_path))
    print(files)
    print()
    combinations = [files[i:i+4] for i in range(len(files)-3)]
    print(combinations)
    return combinations


start_time = time.time()
scrape_historical_data(2008, 2023, True, False)
scrape_historical_data(2008, 2023, False, False)
aggregate_training_data()
print(f"Runtime: {time.time()-start_time:.3f} seconds")