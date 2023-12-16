import os
import time
import pandas as pd

def scrape_historical_data(start_year, end_year, verbose):
    for year in range(start_year, end_year+1):
        filename = f'{year-1}-{year}_skater_data.csv'
        file_path = os.path.join(os.path.dirname(__file__), 'Sim Engine Data', 'Historical Skater Data', filename)

        if os.path.exists(file_path):
            if verbose:
                print(f'{filename} already exists in the following directory: {file_path}')
            continue

        url = f"https://www.naturalstattrick.com/playerteams.php?fromseason={year-1}{year}&thruseason={year-1}{year}&stype=2&sit=all&score=all&stdoi=std&rate=n&team=ALL&pos=S&loc=B&toi=0&gpfilt=none&fd=&td=&tgp=410&lines=single&draftteam=ALL"
        df = pd.read_html(url)[0]
        df = df.iloc[:, 1:]
        if verbose == True:
            print(df)

        export_path = os.path.dirname(file_path)
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        df.to_csv(os.path.join(export_path, f'{filename}.csv'))
        if verbose:
            print(f'{filename}.csv has been downloaded to the following directory: {export_path}')


start_time = time.time()
scrape_historical_data(2008, 2023, True)
print(f"Runtime: {time.time()-start_time:.3f} seconds")