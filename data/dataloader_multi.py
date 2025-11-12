import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
from .datasets_multi import xRIMdataset, bundlexRIM
import glob
import time
import datetime
import functools
import operator
from numpy import argmax

def compute_density(distances, mask, R, path, logfile=None):
    distances = np.ma.masked_where(distances < R, distances) #masked array where elements that distances< R
    density = np.sum(distances.mask, axis=1) # count up the number of masked elements
    density = density / (np.pi * R ** 2)

    density = np.ma.masked_array(density, mask=mask)
    density = density.compressed()
    return density

def compute_pt(pt, myDistrances, mask, R, path, logfile=None):
    myMask = (myDistrances==0) | (myDistrances > R)
    pt_matrix = np.broadcast_to(pt[:, np.newaxis, :], (myDistrances.shape[0], myDistrances.shape[1], myDistrances.shape[1]))
    pt_matrix = np.where(myMask, np.nan, pt_matrix)
    pt_around_track = np.nansum(pt_matrix, axis=-1)
    pt_around_track = np.ma.masked_array(pt_around_track, mask=mask)
    #pt_around_track = np.sum(pt[:, :, np.newaxis] * (distances < R), axis=2)
    #distances_squared_masked = np.ma.masked_where(distances > R, distances)
    ## Calculate the sum of pT around the track
    #pt_around_track = np.sum(pt_matrix * distances_squared_masked.mask, axis=1)
    #pt_around_track = np.ma.masked_array(pt_around_track, mask=mask)
    pt_around_track = pt_around_track.compressed()
    return pt_around_track

def compute_sumDensity(distances, mask, R, path, logfile=None):
    myMask = distances > R
    
    distances = np.where(myMask, np.nan, distances)
    distances = np.nansum(distances, axis=1)
    myDensity = distances / (np.pi * R ** 2)
    myDensity = np.ma.masked_array(myDensity, mask=mask)
    myDensity = myDensity.compressed()
    return myDensity

def get_densities(df_ep, pt, R, path, logfile=None):

    if not isinstance(R, list):
        R = [R]
    mask = df_ep[:, :, 0].mask
    df_ep = np.expand_dims(df_ep, -2) #Negative value: inserted at that position, counting from the right
    df_ep_T = df_ep.transpose(0, 2, 1, 3)
    abs_distance = np.abs(df_ep - df_ep_T) #abs_distance[i,j,k,:] corresponds to the absolute distance between track j and k in event i
    # computes the minimum angular distance between the two particles to ensure the shortest distance 
    abs_distance[:, :, :, 1] = np.minimum(
        2 * np.pi - abs_distance[:, :, :, 1], abs_distance[:, :, :, 1]
    )
    distances = np.sqrt(np.sum(np.power(abs_distance, 2), axis=-1))
    # replaces distances of 0.0 with 999
    myDistances = distances
    distances = np.where(distances == 0.0, distances + 999, distances)
    collect= [compute_density(distances, mask, r, path, logfile) for r in R]
    pt_around_track = [compute_pt(pt, myDistances, mask, r, path, logfile) for r in R]
    sum_collect = [compute_sumDensity(distances, mask, r, path, logfile) for r in R]
    #return collect
    return collect, sum_collect, pt_around_track

def prepxRIMdata(path, logfile=None, isTraining=True, sample="JZ7W"):

    """
    This function looks at the path specified and creates a single h5 file from the csv files in the path.
    
    Parameters:
    ----------
    path (str): Path to the directory containing the csv files: This should be the name of the sample (eg. JZ7W)
    
    Returns
    -------
    newData (pd.DataFrame): A dataframe containing all the data from the csv files in the path.

    """
    
    print("Creating the h5 files.", file=logfile, flush=True)
    t0 = time.time()
    wpath_to_data = f"{path}/MC-overlay_{sample}/"
    wopath_to_data = f"{path}/Track-overlay_{sample}/"

    #Glob all the csvs in the path corresponding to pu or no pu (MC or Track)
    pucsvlist = glob.glob(wpath_to_data + "*.csv")
    wopucsvlist = glob.glob(wopath_to_data + "*.csv")
    print(f"pucsvlist files:{len(pucsvlist)}", file=logfile, flush=True)
    print(f"wopucsvlist files:{len(wopucsvlist)}", file=logfile, flush=True)
    pileup = pd.concat([pd.read_csv(csv) for csv in pucsvlist])
    nopileup = pd.concat([pd.read_csv(csv) for csv in wopucsvlist])

    # print(pileup.columns, file=logfile, flush=True)
    t1 = time.time()
    print(
        f"\ndfs loaded: {str(datetime.timedelta(seconds=round(t1-t0)))}.",
        file=logfile,
        flush=True,
    )

    #Sort the dataframes by the event number - this makes it easier to match the pu and no pu data.
    #Additionally use Px to sort the dataframe - This makes sure that the data is sorted in the same way for both pu and no pu data and the most
    #relavant truth come first. (i.e. those with the highest Px (or maybe alternatitely Pt?))
    #if isTraining:
    pileup = pileup.sort_values(by=["EvtNumber", "Px"])
    pileup["hasMatch"] = pileup["MatchProb"] > 0.5
    pileup.columns = list(pileup.columns[:9]) + [
        "w_" + string for string in pileup.columns.values[9:]
    ]

    phi = np.arctan2(pileup["Py"], pileup["Px"])
    phi = np.where(phi > np.pi, phi - 2 * np.pi, phi)
    phi = np.where(phi <= -np.pi, phi + 2 * np.pi, phi)

    pileup["Phi"] = phi
    #if isTraining:
    nopileup = nopileup.sort_values(by=["EvtNumber", "Px"])
    nopileup["hasMatch"] = nopileup["MatchProb"] > 0.5
    nopileup.columns = list(nopileup.columns[:9]) + [
        "wo_" + string for string in nopileup.columns.values[9:]
    ]
    print(nopileup[["wo_hasMatch"]].head())
    print(nopileup.columns[9:])

    #Get unique eventIDs from the two samples - We would use this to match the pu and no pu data and tally
    puevts = pileup["EvtNumber"].unique()
    wopuevts = nopileup["EvtNumber"].unique()
    #Get the eventIDs that are common to both samples
    matchedevents = np.intersect1d(puevts, wopuevts)
    #Get the truth data for the matched events
    pileup = pileup[pileup["EvtNumber"].isin(matchedevents)]
    nopileup = nopileup[nopileup["EvtNumber"].isin(matchedevents)]
    print(f"Total matched events in two samples: {matchedevents.shape}", file=logfile, flush=True)

    assert matchedevents.shape[0] > 0, "No matched events found. Please check the data."


    # Make sure the events are all lined up. This is important for the matching of the pu and no pu data.
    assert (
        nopileup["EvtNumber"].values - pileup["EvtNumber"].values
    ).sum() == 0, "Events Misalligned. Check CSV creation/AODs"
    assert (
        nopileup["PdgID"].values - pileup["PdgID"].values
    ).sum() == 0, "Truths Misalligned. Check CSV creation"
    '''assert (
        nopileup["Px"].values - pileup["Px"].values
    ).sum() == 0, "Truths Misalligned. Check CSV creation"
    '''
    
    # Now start the splicing process.

    pileup.reset_index(drop=True, inplace=True)
    nopileup.reset_index(drop=True, inplace=True)
    # Make a new df with the truth data as the base and add the relevant pu and no pu (MC and Track) data to it.
    # The truth data is the same for both pu and no pu data so we can just that from the pu data.
    newData = pd.concat(
        [
            pileup[
                [
                    "EvtNumber",
                    "PdgID",
                    "Px",
                    "Py",
                    "Pz",
                    "E",
                    "Pt",
                    "Eta",
                    "Phi",
                    "w_Mass",
                    #"w_leadJetpT",
                    #"w_nleadJetpT",
                    "w_numPU",
                    "w_numvtx",
                    "w_MatchProb",
                    "w_hasMatch",
                ]
            ],
            nopileup[
                [
                    #"wo_leadJetpT",
                    #"wo_nleadJetpT",
                    "wo_numPU",
                    "wo_numvtx",
                    "wo_MatchProb",
                    "wo_hasMatch",
                ]
            ],
        ],
        axis=1,
    )
    # Mask tracks from muons
    #mask = (newData['PdgID'] == 13) | (newData['PdgID'] == -13)
    #newData = newData[~mask]
    # The classifier needs to learn the target label.
    # It is defined as positive/label 1 if the particle has a match in the no pu (Track-Overlay) sample but not in the pu (MC overlay) sample.
    #newData["TargetLabel"] = newData["wo_hasMatch"] & np.logical_not(
    #    newData["w_hasMatch"]
    #)
    #True(1) if cases where Track Overlay has a match, but MC Overlay does not OR cases where Track Overlay itself has no match, regardless of MC Overlay
    #newData["TargetLabel"] = newData["wo_hasMatch"] & ~newData["w_hasMatch"] | ~newData["wo_hasMatch"]
    #True(1) if cases where Track Overlay has a match, but MC Overlay does not OR Track Overlay has no match
    newData["TargetLabel"] = (newData["wo_hasMatch"] & ~newData["w_hasMatch"]) | (~newData["wo_hasMatch"] & newData["w_hasMatch"])

    newData["TruthMultiplicity"] = newData.groupby("EvtNumber")["EvtNumber"].transform(
        "count"
    )
    newData["TruthPhi"] = np.arctan(newData["Py"] / newData["Px"])
    newData["eventPx"] = newData.groupby("EvtNumber")["Px"].transform("sum")
    newData["eventPy"] = newData.groupby("EvtNumber")["Py"].transform("sum")
    newData["eventPt"] = (newData["eventPx"] ** 2 + newData["eventPy"] ** 2) ** 0.5
    numEvents = len(newData["EvtNumber"].unique())
    newData = newData.set_index("EvtNumber") #EvtNumber column is used as the index corresponding to a separate group of rows
    # Split the original DataFrame into groups based on the EvtNumber index
    newData = newData.set_index(newData.groupby("EvtNumber").cumcount(), append=True)
    newData_eponly = newData[["Eta", "Phi"]]
    newData_pt = newData[["Pt"]]
    # newData_eponly.to_xarray() converts the dataset based on the dim of the indices of the newData_eponly 
    ep_only = newData_eponly.to_xarray().to_array().values # ep_only = (2, nEvents, nTracks). 2: Eta, Phi
    ep_only = np.transpose(ep_only, axes=[1, 2, 0]) # the resulting shape is (nEvents, nTracks, 2)
    ep_only = np.ma.masked_invalid(ep_only)
    pt_only = newData_pt.to_xarray().to_array().values
    pt_only= np.transpose(pt_only, axes=[1, 2, 0])
    pt_only = np.squeeze(pt_only, axis=-1)
    pt_only = np.ma.masked_invalid(pt_only)
    t3 = time.time()
    
    # Create a few density based variables as well.
    # A higher density at a smaller radius indicates that there are more tracks in close proximity to each other.
    # It is possible that tracks are distributed more evenly across a larger area
    #print("Calculating densities.", file=logfile, flush=True)
    nchunk = 1000
    start = 0
    density_02 = []
    density_05 = []
    sum_density_02 = []
    sum_density_05 = []
    scalar_pT_02 = []
    scalar_pT_05 = []
    while start < numEvents:
        print(f"Evaluating event {start}-{start+nchunk}")
        collect, sum_collect, pt_around_track= get_densities(ep_only[start : start + nchunk], pt_only[start: start+nchunk], [0.2, 0.05], path, logfile) 
        #collect, sum_collect = get_densities(ep_only[start : start + nchunk], [0.2, 0.05], path, logfile) 
        density_02.append(collect[0]) #too see if more tracks are within that radius
        density_05.append(collect[1]) 
        sum_density_02.append(sum_collect[0])
        sum_density_05.append(sum_collect[1])
        #scalar_sum_pT_d2, scalar_sum_pT_d5 = get_scalarSumpT(distances, pt_only[start : start + nchunk], [0.2, 0.05], path, logfile)
        scalar_pT_02.append(pt_around_track[0])
        scalar_pT_05.append(pt_around_track[1])
        start += nchunk
    t4 = time.time()
    print(
        f"Densities computed {str(datetime.timedelta(seconds=round(t4-t3)))} s.",
        file=logfile,
        flush=True,
    )
    density_02 = functools.reduce(operator.iconcat, density_02, []) #len(density_02) = number of tracks
    density_05 = functools.reduce(operator.iconcat, density_05, [])
    sum_density_02 = functools.reduce(operator.iconcat, sum_density_02, [])
    sum_density_05 = functools.reduce(operator.iconcat, sum_density_05, [])
    scalar_pT_02 = functools.reduce(operator.iconcat,scalar_pT_02, [])
    scalar_pT_05 = functools.reduce(operator.iconcat, scalar_pT_05, [])
    newData["R02"] = density_02 # the list of all tracks pairs values for each event be assigned as a R02 column
    newData["R05"] = density_05
    newData["sumR02"] = sum_density_02 # the list of all tracks pairs values for each event be assigned as a R02 column
    newData["sumR05"] = sum_density_05
    newData["sumpT02"] = scalar_pT_02
    newData["sumpT05"] = scalar_pT_05
    #scalar_pT_02 = functools.reduce(operator.iconcat, scalar_pT_02, [])
    #scalar_pT_05 = functools.reduce(operator.iconcat, scalar_pT_05, [])
    #newData["sPT02"] = scalar_pT_02
    #newData["sPT05"] = scalar_pT_05
    # Save the data to a h5 file for later use.
    newData.to_hdf(f"{path}/{sample}.h5", key="newData", format="t")
    print(
        f"h5 saved, moving on. {str(datetime.timedelta(seconds=round(t4-t0)))}",
        file=logfile,
        flush=True,
    )
    
    return newData


def prepareData(path, logfile=None, sample="JZ7W", use_densities=True):
    """
    Load and prepare data with all features. This is the main data preparation function
    that should be called once to create processed data files.
    
    Parameters
    ----------
    path : str
        Path to the directory containing the raw data.
    logfile : file, optional
        File to write the log to.
    sample : str
        Which sample to load (e.g., "JZ7W", "ttbar").
    use_densities : bool
        Whether to compute and include density-based features.
    
    Returns
    -------
    fullData_matched : pd.DataFrame
        Matched class (TargetLabel=1, MatchProb > 0.5) samples.
    fullData_unmatched : pd.DataFrame
        Unmatched class (TargetLabel=0, MatchProb <= 0.5) samples.
    feat_list : list
        List of feature names.
    """
    existinghpath_matched = glob.glob(f"{path}/matched_{sample}_data*.h5")
    existinghpath_unmatched = glob.glob(f"{path}/unmatched_{sample}_data*.h5")
    
    if existinghpath_matched and existinghpath_unmatched:
        print(f"Loading existing processed data from {path}", file=logfile, flush=True)
        fullData_matched = pd.read_hdf(existinghpath_matched[0])
        fullData_unmatched = pd.read_hdf(existinghpath_unmatched[0])
        print(f"Loaded matched: {fullData_matched.shape}, unmatched: {fullData_unmatched.shape}", file=logfile, flush=True)
    else:
        print("No existing processed files found. Creating from raw CSV files...", file=logfile, flush=True)
        fullData = prepxRIMdata(path, logfile, sample=sample, isTraining=True)
        fullData_unmatched = fullData[fullData["TargetLabel"] == 0]
        fullData_matched = fullData[fullData["TargetLabel"] == 1]
        
        # Save processed data
        fullData_unmatched.to_hdf(f"{path}/unmatched_{sample}_data.h5", key="data")
        fullData_matched.to_hdf(f"{path}/matched_{sample}_data.h5", key="data")
        print(f"Saved processed data: matched_{sample}_data.h5, unmatched_{sample}_data.h5", file=logfile, flush=True)
    
    # Create a sample dataset to get the feature list
    sample_data = pd.concat([fullData_matched.head(100), fullData_unmatched.head(100)])
    xRIM_sample = xRIMdataset(sample_data, use_densities=use_densities, logfile=logfile)
    feat_list = xRIM_sample.featureList
    
    return fullData_matched, fullData_unmatched, feat_list


def loadTrainTestData(path, logfile=None, sample="JZ7W", split=0.8, use_densities=True, 
                      pos_sample_size=None, neg_sample_size=None):
    """
    Load processed data and split into training and test sets with class balancing.
    
    Parameters
    ----------
    path : str
        Path to the directory containing processed h5 files.
    logfile : file, optional
        File to write the log to.
    sample : str
        Which sample to load.
    split : float
        Fraction of data to use for training (rest is test).
    use_densities : bool
        Whether to use density-based features.
    pos_sample_size : int, optional
        Number of matched samples to use. If None, uses all available.
    neg_sample_size : int, optional
        Number of unmatched samples to use. If None, uses all available.
    
    Returns
    -------
    F_train : np.ndarray
        Training features.
    F_test : np.ndarray
        Test features.
    L_train : np.ndarray
        Training labels.
    L_test : np.ndarray
        Test labels.
    W_train : np.ndarray
        Training sample weights (for class balancing).
    W_test : np.ndarray
        Test sample weights.
    feat_list : list
        List of feature names.
    """
    existinghpath_matched = glob.glob(f"{path}/matched_{sample}_data*.h5")
    existinghpath_unmatched = glob.glob(f"{path}/unmatched_{sample}_data*.h5")
    
    if not (existinghpath_matched and existinghpath_unmatched):
        raise FileNotFoundError(
            f"Processed data not found at {path}. Please run prepareData() first."
        )
    
    fullData_matched = pd.read_hdf(existinghpath_matched[0])
    fullData_unmatched = pd.read_hdf(existinghpath_unmatched[0])
    
    # Sample if sizes specified
    if pos_sample_size is not None:
        fullData_matched = fullData_matched.sample(n=min(pos_sample_size, len(fullData_matched)), random_state=42)
    if neg_sample_size is not None:
        fullData_unmatched = fullData_unmatched.sample(n=min(neg_sample_size, len(fullData_unmatched)), random_state=42)
    
    print(f"Using matched: {fullData_matched.shape}, unmatched: {fullData_unmatched.shape}", file=logfile, flush=True)
    
    # Combine and create dataset
    Data_concat = pd.concat([fullData_matched, fullData_unmatched], axis=0)
    xRIM_train = xRIMdataset(Data_concat, use_densities=use_densities, logfile=logfile)
    
    feat = xRIM_train.features
    labels = xRIM_train.labels
    feat_list = xRIM_train.featureList
    
    # Compute class weights for imbalanced data
    false_classW = np.full(fullData_unmatched.shape[0], 0)
    true_classW = np.full(fullData_matched.shape[0], 1)
    mixed_Y = np.concatenate([true_classW, false_classW])
    unique_classes = np.unique(mixed_Y)
    unique_y = [0., 1.]
    
    classWeight = dict(zip(
        unique_y, 
        sklearn.utils.class_weight.compute_class_weight('balanced', classes=unique_classes, y=mixed_Y)
    ))
    
    a = list(classWeight.values())
    print(f"Class weights: NoDiff={a[0]:.4f}, Diff={a[1]:.4f}", file=logfile, flush=True)
    
    # Assign weights to each sample
    W1 = np.where(labels == 0, a[0], labels)
    W = np.where(labels == 1, a[1], W1)
    
    # Train/test split
    F_train, F_test, L_train, L_test, W_train, W_test = train_test_split(
        feat, labels, W, test_size=1-split, shuffle=True, random_state=1
    )
    
    print(f"Split complete - Train: {F_train.shape}, Test: {F_test.shape}", file=logfile, flush=True)
    
    return F_train, F_test, L_train, L_test, W_train, W_test, feat_list


def loadValidationData(path, logfile=None, sample="JZ7W", use_densities=True):
    """
    Load data for validation/evaluation only (no train/test split).
    
    Parameters
    ----------
    path : str
        Path to the directory containing processed h5 files.
    logfile : file, optional
        File to write the log to.
    sample : str
        Which sample to load.
    use_densities : bool
        Whether to use density-based features.
    
    Returns
    -------
    bundle : bundlexRIM
        Bundle containing validation dataset.
    vali_features : np.ndarray
        Validation features.
    mixed : xRIMdataset
        Combined dataset.
    """
    existinghpath_matched = glob.glob(f"{path}/matched_{sample}_data*.h5")
    existinghpath_unmatched = glob.glob(f"{path}/unmatched_{sample}_data*.h5")
    
    if not (existinghpath_matched and existinghpath_unmatched):
        raise FileNotFoundError(
            f"Processed data not found at {path}. Please run prepareData() first."
        )
    
    fullData_matched = pd.read_hdf(existinghpath_matched[0])
    fullData_unmatched = pd.read_hdf(existinghpath_unmatched[0])
    
    Data_concat = pd.concat([fullData_matched, fullData_unmatched], axis=0)
    mixed = xRIMdataset(Data_concat, use_densities=use_densities, logfile=logfile)
    bundle = bundlexRIM(validation=mixed, logfile=logfile)
    vali_features = mixed.features
    
    return bundle, vali_features, mixed


def loadEvaluationData(path, logfile=None, sample="JZ7W", use_densities=True, 
                       pos_sample_size=None, neg_sample_size=None):
    """
    Load data for evaluation/inference. Returns features and labels without train/test split.
    Use this when you want to evaluate on a complete dataset (can be new data).
    
    Parameters
    ----------
    path : str
        Path to the directory containing processed h5 files.
    logfile : file, optional
        File to write the log to.
    sample : str
        Which sample to load.
    use_densities : bool
        Whether to use density-based features.
    pos_sample_size : int, optional
        Number of matched samples to use. If None, uses all available.
    neg_sample_size : int, optional
        Number of unmatched samples to use. If None, uses all available.
    
    Returns
    -------
    F_eval : np.ndarray
        Evaluation features.
    L_eval : np.ndarray
        Evaluation labels.
    feat_list : list
        List of feature names.
    df_eval : pd.DataFrame
        Full dataframe for event-level evaluation.
    """
    existinghpath_matched = glob.glob(f"{path}/matched_{sample}_data*.h5")
    existinghpath_unmatched = glob.glob(f"{path}/unmatched_{sample}_data*.h5")
    
    if not (existinghpath_matched and existinghpath_unmatched):
        raise FileNotFoundError(
            f"Processed data not found at {path}. Please run prepareData() first."
        )
    
    fullData_matched = pd.read_hdf(existinghpath_matched[0])
    fullData_unmatched = pd.read_hdf(existinghpath_unmatched[0])
    
    # Sample if sizes specified
    if pos_sample_size is not None:
        fullData_matched = fullData_matched.sample(n=min(pos_sample_size, len(fullData_matched)), random_state=42)
    if neg_sample_size is not None:
        fullData_unmatched = fullData_unmatched.sample(n=min(neg_sample_size, len(fullData_unmatched)), random_state=42)
    
    print(f"Evaluation data: pos={fullData_matched.shape[0]}, neg={fullData_unmatched.shape[0]}", file=logfile, flush=True)
    
    # Combine and create dataset
    Data_concat = pd.concat([fullData_matched, fullData_unmatched], axis=0)
    xRIM_eval = xRIMdataset(Data_concat, use_densities=use_densities, logfile=logfile)
    
    F_eval = xRIM_eval.features
    L_eval = xRIM_eval.labels
    feat_list = xRIM_eval.featureList
    
    print(f"Evaluation features shape: {F_eval.shape}", file=logfile, flush=True)
    
    return F_eval, L_eval, feat_list, Data_concat


# Legacy function for backward compatibility
def loadData(path, logfile=None, preProcess=True, sample="JZ7W", split=0.8, use_densities=True, isTraining=True):

    """
    DEPRECATED: Use prepareData(), loadTrainTestData(), or loadValidationData() instead.
    
    This function loads the data from the h5 file (if it exists, otherwise it creates it) and returns the data
    as a xRIM dataset.

    Parameters
    ----------
    path : str
        Path to the directory containing the h5 file.

    logfile : file, optional
        File to write the log to. The default is None.

    preProcess : bool, optional
        Whether to preprocess the data or not. The default is True.

    sample : str, optional
        Which sample to load. The default is "JZ7W".

    split : float, optional
        The fraction of the data to use for training. The default is 0.7.

    use_densities : bool, optional
        Whether to use the density based variables or not. The default is True.


    Returns
    -------
    xRIM dataset bundle or train/test split
    """
    #existingh5path = glob.glob(f"{path}/matched_J7_2k_data.h5")
    
    existinghpath_matched = glob.glob(f"{path}/matched_{sample}_data*.h5")
    existinghpath_unmatched = glob.glob(f"{path}/unmatched_{sample}_data*.h5")
    print("existinghpath_matched:", existinghpath_matched)
    if existinghpath_matched:
        #fullData = pd.read_hdf(existingh5path[0])
        fullData_matched = pd.read_hdf(existinghpath_matched[0])
        fullData_unmatched = pd.read_hdf(existinghpath_unmatched[0])
        #fullData_matched = fullData_matched.sample(n=fullData_matched.shape[0], random_state=42)
        #fullData_unmatched = fullData_unmatched.sample(n=fullData_unmatched.shape[0], random_state=42)
        fullData_matched = fullData_matched.sample(n=72744, random_state=42)
        fullData_unmatched = fullData_unmatched.sample(n=872744, random_state=42) 
        # Print the shapes to confirm
        print("Shape of fullData_matched:", fullData_matched.shape)
        print("Shape of fullData_unmatched:", fullData_unmatched.shape)
    else:
       print(
          "No existing npz files found. Creating one from the existing csv files. This will take a moment.",
          file=logfile,
          flush=True,
       )
    
       fullData = prepxRIMdata(path, logfile, sample=sample, isTraining=isTraining)
       fullData_unmatched, fullData_matched = (
       fullData[fullData["TargetLabel"] == 0],
       fullData[fullData["TargetLabel"] == 1],
       ) #pos: Diff(TargetLabel=1), neg: No Diff(TargetLabel=0)
       # Save pos and neg to separate .h5 files
       fullData_unmatched.to_hdf(f"{path}/unmatched_{sample}_data.h5", key="data")
       fullData_matched.to_hdf(f"{path}/matched_{sample}_data.h5", key="data")
    print("pos/Diff shape:", fullData_matched.shape, file=logfile, flush=True)
    print("neg/NoDiff shape:", fullData_unmatched.shape, file=logfile, flush=True)
    #fullData6 = fullData6.sample(n=224871, random_state=42) #J7
    #fullData12 = fullData12.sample(n=324871, random_state=42) #J7
    Data_concat = pd.concat([fullData_matched, fullData_unmatched], axis=0)
    #pos = pd.concat([fullData1, fullData2, fullData3, fullData4, fullData5, fullData6], axis=0)
    #neg = pd.concat([fullData7, fullData8, fullData9, fullData10, fullData11, fullData12], axis=0)
    #pos = pd.concat([fullData6], axis=0)
    #neg = pd.concat([fullData12], axis=0)
    '''
    neg, pos = (
       fullData[fullData["TargetLabel"] == 0],
       fullData[fullData["TargetLabel"] == 1],
    ) #pos: Diff, neg: No Diff
    '''
    # Save pos and neg to separate .h5 files
    #neg.to_hdf("neg_data.h5", key="data")
    #pos.to_hdf("pos_data.h5", key="data")
    if not isTraining:
       
       #mixed = pd.concat([pos, neg])
       #Data_concat.sample = Data_concat.sample(frac=1, random_state=1)
       mixed = xRIMdataset(Data_concat, use_densities=use_densities, logfile=logfile )
       bundle = bundlexRIM(validation=mixed, logfile=logfile)
       vali_features = mixed.features
       #vali_conditions = bundle.validation.condFeatures
       
       # Select 10k events of label 0
       #neg = fullData[fullData["TargetLabel"] == 0].sample(n=207750, random_state=42)
       #pos = fullData[fullData["TargetLabel"] == 1].sample(n=200000, random_state=42)
       #This at the moment looks at the # of events in pu (MC) and no PU (Track Overlay) samples and takes the smaller number.
       #! TODO: Make the classifier training with weights so that the number of events in the pu and no pu samples can be different.
       #size = min(pos.shape[0], neg.shape[0])
    else:
       #fullData_matched_train, fullData_matched_test = train_test_split(fullData_matched, test_size=1-split, random_state=42)
       #fullData_unmatched_train, fullData_unmatched_test = train_test_split(fullData_unmatched, test_size=1-split, random_state=42)
       #Data_train = pd.concat([fullData_matched_train, fullData_unmatched_train], axis=0)
       #Data_test = pd.concat([fullData_matched_test, fullData_unmatched_test], axis=0)
       #Data_test = Data_test.sample(frac=1, random_state=1)
       xRIM_train = xRIMdataset(Data_concat, use_densities=use_densities, logfile=logfile )
       #xRIM_test = xRIMdataset(Data_test,  use_densities=use_densities, logfile=logfile)
       #feat = np.concatenate([xRIM_J4.features, xRIM_J7.features])
       #cond = np.concatenate([xRIM_J4.condFeatures,  xRIM_J7.condFeatures])
       #lebales = np.concatenate([xRIM_J4.labels, xRIM_J7.labels])
       #feat = np.concatenate([xRIM_J2.features, xRIM_J3.features, xRIM_J7.features]) #J4 + J7
       #cond = np.concatenate([xRIM_J2.condFeatures, xRIM_J3.condFeatures, xRIM_J7.condFeatures])
       #lebales = np.concatenate([xRIM_J2.labels, xRIM_J3.labels, xRIM_J7.labels])
       feat = xRIM_train.features
       #cond = xRIM_J7.condFeatures
       lebales = xRIM_train.labels
       feat_list = xRIM_train.featureList
       #cond_list = xRIM_J7.condFeatureList
       false_classW = np.full(fullData_unmatched.shape[0], 0) #Array of 0s for NoDiff class
       true_classW = np.full(fullData_matched.shape[0], 1) #Array of 1s for Diff class
       mixed_Y = np.concatenate([true_classW, false_classW])
       unique_classes = np.unique(mixed_Y)
       unique_y = [0.,1.]
       #computing class weights to handle class imbalance
       classWeight = dict(zip(unique_y, sklearn.utils.class_weight.compute_class_weight('balanced', classes=unique_classes, y=mixed_Y)))

       #classWeight = dict(zip(unique_y,sklearn.utils.class_weight.compute_class_weight('balanced', unique_classes, mixed_Y)))
       a = list(classWeight.values()) #a[0] is the weight for NoDiff class, and a[1] is the weight for Diff class.
       W1 = np.where(lebales==0, a[0], lebales) #Assign weight a[0] to NoDiff class 
       W = np.where(lebales==1, a[1], W1) #Assign weight a[1] to Diff class
       #pos_ClassW = np.full(fullData12.shape[0], a[1])
       #neg_ClassW = np.full(fullData_matched.shape[0], a[0])
       #C = np.concatenate( (pos_ClassW, neg_ClassW) )
       #W = np.repeat(C, 3)
       F_train, F_test, L_train, L_test, W_train, W_test= train_test_split(feat, lebales, W, test_size=1-split, shuffle=True,random_state=1)
       #bundle = bundlexRIM(validation=xRIM_test, logfile=logfile)
    #if preProcess:
    #   bundle.preProcess()
    if not isTraining:
       return bundle, vali_features, mixed
    else:
       #return bundle, feat, lebales, W, feat_list
       return F_train, F_test,L_train, L_test, W_train, W_test, feat_list
       #return bundle, myValiset, W_test
