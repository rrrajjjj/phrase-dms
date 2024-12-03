import matplotlib.pyplot as plt
import os
import urllib.parse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
from LogParser import LogParser

parser = LogParser("../logs")
# Get the list of patient IDs
patient_ids = parser.get_patient_ids()

# Initialize an empty data structure to hold data per protocol, per game mode, per user
protocols = {}

# Loop through each patient and collect their data
for patient_id in patient_ids:
    patient_dms = parser.get_dms(patient_id)
    patient_hits_errors = parser.get_hits_errors(patient_id)
    for protocol in patient_dms.keys():
        if protocol not in protocols:
            protocols[protocol] = {}
        for game_mode in patient_dms[protocol].keys():
            if game_mode not in protocols[protocol]:
                protocols[protocol][game_mode] = {}
            # Store the patient's DM data and hits/errors data for this game mode
            dms = patient_dms[protocol][game_mode]
            hits_errors = patient_hits_errors.get(protocol, {}).get(game_mode, [])
            protocols[protocol][game_mode][patient_id] = {
                'dms': dms,
                'hits_errors': hits_errors
            }

# Now, 'protocols' is a nested dictionary:
# protocols[protocol][game_mode][patient_id] = {'dms': dms_dict, 'hits_errors': hits_errors_list}

# Create output directories for images
output_dir = '../plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

redundancy_output_dir = '../redundancy_analysis'
if not os.path.exists(redundancy_output_dir):
    os.makedirs(redundancy_output_dir)

# Function to sanitize filenames by removing or replacing invalid characters
def sanitize_filename(name):
    import string
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    sanitized = ''.join(c if c in valid_chars else '_' for c in name)
    return sanitized.replace(' ', '_')

# Function to compute sliding average
def sliding_average(data, window_size):
    if len(data) < window_size:
        return np.array(data)  # Return the original data if too short
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Open the HTML file for writing
html_filename = '../index.html'
with open(html_filename, 'w') as html_file:
    # Write the initial HTML headers and styles
    html_file.write('<html><head><title>Plots</title>\n')

    # Include CSS styles for tabs and tables
    html_file.write('''
    <style>
    /* Style the tab */
    .tab {
      overflow: hidden;
      border-bottom: 1px solid #ccc;
    }

    /* Style the buttons inside the tab */
    .tab button {
      background-color: inherit;
      float: left;
      border: none;
      outline: none;
      cursor: pointer;
      padding: 10px 12px;
      transition: 0.3s;
      font-size: 16px;
    }

    /* Change background color of buttons on hover */
    .tab button:hover {
      background-color: #ddd;
    }

    /* Create an active/current tablink class */
    .tab button.active {
      background-color: #ccc;
    }

    /* Style the tab content */
    .tabcontent {
      display: none;
      padding: 10px 0;
    }

    /* Style for tables */
    table {
      border-collapse: collapse;
      width: 100%;
    }

    table, th, td {
      border: 1px solid #ccc;
    }

    th, td {
      text-align: center;
      padding: 8px;
    }

    /* Style for DM Legend */
    .dm-legend {
      text-align: left;
      margin-top: 10px;
    }
    .dm-legend ul {
      list-style-type: none;
      padding: 0;
    }
    .dm-legend li {
      margin: 2px 0;
    }

    </style>
    </head><body>
    ''')

    # Include JavaScript functions for tabs
    html_file.write('''
    <script>
    function openTab(evt, tabName) {
      var i, tabcontent, tablinks;

      // Hide all tab contents
      tabcontent = document.getElementsByClassName("tabcontent");
      for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
      }

      // Remove 'active' class from all tab links
      tablinks = document.getElementsByClassName("tablinks");
      for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
      }

      // Show the selected tab content and add 'active' class to the clicked tab
      document.getElementById(tabName).style.display = "block";
      evt.currentTarget.className += " active";
    }

    // Automatically click the first tab to display it on page load
    document.addEventListener('DOMContentLoaded', function() {
      document.querySelector('.tablinks').click();
    });
    </script>
    ''')

    # Create the tab buttons
    html_file.write('''
    <div class="tab">
      <button class="tablinks" onclick="openTab(event, 'DMPlots')">DM Plots</button>
      <button class="tablinks" onclick="openTab(event, 'RedundancyAnalysis')">Redundancy Analysis</button>
    </div>
    ''')

    # Start DM Plots tab content
    html_file.write('<div id="DMPlots" class="tabcontent">\n')

    # Generate DM Plots content
    for protocol in protocols.keys():
        html_file.write(f'<h1>{protocol}</h1>\n')
        html_file.write('<table>\n')  # Start a table

        game_modes = protocols[protocol]
        # First, write the header row with patient IDs
        html_file.write('<tr><th>Game Mode</th>')
        for idx, patient_id in enumerate(patient_ids):
            html_file.write(f'<th>User {patient_id}</th>')
        html_file.write('</tr>\n')

        # Loop through game modes
        for game_mode in game_modes.keys():
            html_file.write(f'<tr><td>{game_mode}</td>\n')  # Start a table row

            # For tracking if it's the first user in the row
            is_first_patient = True

            # For each patient ID
            for patient_id in patient_ids:
                # Check if this patient has data for this game mode
                if patient_id in protocols[protocol][game_mode]:
                    data = protocols[protocol][game_mode][patient_id]
                    dms = data['dms']
                    hits_errors = data['hits_errors']
                    if not dms:
                        html_file.write('<td>No Data</td>\n')
                        continue  # Skip if no data

                    # Create figure with two subplots
                    fig = plt.figure(figsize=(4, 4.5))
                    gs = fig.add_gridspec(3, 1)

                    # Top 2/3: DM Plots
                    ax1 = fig.add_subplot(gs[:2, 0])
                    for dm_name, values in dms.items():
                        ax1.plot(values, label=dm_name)
                    if is_first_patient:
                        ax1.legend(fontsize='small')
                    else:
                        ax1.legend().set_visible(False)
                    ax1.set_title(f'{game_mode} - User {patient_id}', fontsize=10)
                    ax1.set_ylabel('DM Value')

                    # Bottom 1/3: Success Rate Plot
                    ax2 = fig.add_subplot(gs[2, 0], sharex=ax1)
                    if hits_errors:
                        # Apply sliding average before resizing
                        success_rate = sliding_average(hits_errors, window_size=10)
                        # Resize success_rate to match length of DMs
                        total_dm_length = len(next(iter(dms.values())))
                        success_rate_resized = np.interp(
                            np.linspace(0, len(success_rate) - 1, total_dm_length),
                            np.arange(len(success_rate)),
                            success_rate
                        )
                        ax2.plot(success_rate_resized, color='green')
                        ax2.set_ylabel('Success Rate')
                        ax2.set_xlabel('Time')
                        ax2.set_ylim(0, 1.1)
                        # Draw horizontal dotted line at y=0.7
                        ax2.axhline(y=0.7, color='red', linestyle='--', linewidth=1)

                    else:
                        ax2.text(0.5, 0.5, 'No Hits/Errors Data', horizontalalignment='center',
                                 verticalalignment='center', transform=ax2.transAxes)
                        ax2.set_ylabel('Success Rate')
                        ax2.set_xlabel('Time')
                        ax2.set_ylim(0, 1)
                    plt.tight_layout()

                    # Save the plot as an image
                    # Sanitize filenames
                    safe_protocol = sanitize_filename(protocol)
                    safe_game_mode = sanitize_filename(game_mode)
                    safe_patient_id = sanitize_filename(str(patient_id))
                    image_filename = f'{safe_protocol}_{safe_game_mode}_{safe_patient_id}.png'
                    image_path = os.path.join(output_dir, image_filename)
                    plt.savefig(image_path)
                    plt.close()

                    # Include the image in the HTML file
                    # Use os.path.relpath to get the relative path from the HTML file to the image
                    relative_image_path = os.path.relpath(image_path, os.path.dirname(html_filename))
                    image_src = urllib.parse.quote(relative_image_path.replace('\\', '/'), safe='/')
                    html_file.write(f'<td><img src="{image_src}" alt="User {patient_id}" width="200"></td>\n')
                else:
                    # Patient does not have data for this game mode
                    html_file.write('<td>No Data</td>\n')

                # After processing the first patient, set the flag to False
                is_first_patient = False

            html_file.write('</tr>\n')  # End the table row

        html_file.write('</table>\n')  # End the table

    # End DM Plots tab content
    html_file.write('</div>\n')

    # Start Redundancy Analysis tab content
    html_file.write('<div id="RedundancyAnalysis" class="tabcontent">\n')

    # Generate Redundancy Analysis content (same as before)
    for protocol, game_modes in protocols.items():
        html_file.write(f'<h1>{protocol}</h1>\n')
        html_file.write('<table>\n')  # Start a table

        # Write the header row
        html_file.write('<tr><th>Game Mode</th>')
        html_file.write('<th>Correlation Matrix</th>')
        html_file.write('<th>DM Legend</th>')
        html_file.write('<th>PCA Scree Plot</th>')
        html_file.write('</tr>\n')

        for game_mode, user_data in game_modes.items():
            # Collect all DMs for this game mode across users
            aggregated_dms = {}
            for patient_id, data in user_data.items():
                dms = data['dms']
                for dm_name, values in dms.items():
                    if dm_name not in aggregated_dms:
                        aggregated_dms[dm_name] = []
                    aggregated_dms[dm_name].extend(values)

            if not aggregated_dms:
                continue  # Skip if no data

            # Ensure all DMs have the same length
            min_length = min(len(v) for v in aggregated_dms.values())
            aggregated_dms = {k: v[:min_length] for k, v in aggregated_dms.items()}

            # Create DataFrame
            df = pd.DataFrame(aggregated_dms)

            # Drop any DMs that cannot be converted to numeric
            df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1)

            if df.empty or df.shape[1] < 2:
                continue  # Need at least two DMs for redundancy analysis

            # Create mapping from original DM names to 'dm1', 'dm2', etc.
            dm_names = df.columns.tolist()
            dm_labels = [f'dm{i+1}' for i in range(len(dm_names))]
            dm_mapping = dict(zip(dm_labels, dm_names))

            # Rename columns in the DataFrame for plotting
            df_renamed = df.copy()
            df_renamed.columns = dm_labels

            # --- Correlation Matrix ---
            corr_matrix = df_renamed.corr()

            # Save correlation matrix heatmap
            plt.figure(figsize=(4, 3))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                        xticklabels=dm_labels, yticklabels=dm_labels)
            plt.title(f'Correlation Matrix\n{protocol} - {game_mode}')
            plt.tight_layout()

            # Sanitize filenames
            safe_protocol = sanitize_filename(protocol)
            safe_game_mode = sanitize_filename(game_mode)
            corr_filename = f'{safe_protocol}_{safe_game_mode}_correlation.png'
            corr_image_path = os.path.join(redundancy_output_dir, corr_filename)
            plt.savefig(corr_image_path)
            plt.close()

            # --- DM Legend ---
            # Create HTML list for the DM legend
            dm_legend_html = '<div class="dm-legend"><ul>'
            for label, name in dm_mapping.items():
                dm_legend_html += f'<li><strong>{label}</strong>: {name}</li>'
            dm_legend_html += '</ul></div>'

            # --- PCA Scree Plot ---
            pca = PCA()
            pca.fit(df.dropna())
            explained_variance = pca.explained_variance_ratio_

            # Save Scree Plot
            num_components = len(explained_variance)
            cumulative_variance = np.cumsum(explained_variance)
            plt.figure(figsize=(4, 3))
            plt.bar(range(1, num_components + 1), explained_variance, alpha=0.5, align='center',
                    label='Individual Explained Variance')
            plt.step(range(1, num_components + 1), cumulative_variance, where='mid',
                     label='Cumulative Explained Variance')
            plt.ylabel('Explained Variance Ratio')
            plt.xlabel('Principal Components')
            plt.title(f'PCA Scree Plot\n{protocol} - {game_mode}')
            plt.legend(loc='best', fontsize='small')
            plt.tight_layout()

            pca_filename = f'{safe_protocol}_{safe_game_mode}_pca.png'
            pca_image_path = os.path.join(redundancy_output_dir, pca_filename)
            plt.savefig(pca_image_path)
            plt.close()

            # Collect image paths
            corr_relative_path = os.path.relpath(corr_image_path, os.path.dirname(html_filename))
            corr_image_src = urllib.parse.quote(corr_relative_path.replace('\\', '/'), safe='/')

            pca_relative_path = os.path.relpath(pca_image_path, os.path.dirname(html_filename))
            pca_image_src = urllib.parse.quote(pca_relative_path.replace('\\', '/'), safe='/')

            # --- Write to HTML ---
            html_file.write('<tr>\n')
            html_file.write(f'<td>{game_mode}</td>\n')
            html_file.write(f'<td><img src="{corr_image_src}" alt="Correlation Matrix" width="250"></td>\n')
            html_file.write(f'<td>{dm_legend_html}</td>\n')
            html_file.write(f'<td><img src="{pca_image_src}" alt="PCA Scree Plot" width="250"></td>\n')
            html_file.write('</tr>\n')

        html_file.write('</table>\n')  # End the table

    # End Redundancy Analysis tab content
    html_file.write('</div>\n')

    # Write the closing HTML tags
    html_file.write('</body></html>\n')
