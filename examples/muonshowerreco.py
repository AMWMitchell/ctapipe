import sys
import pyhessio
import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import Distance
import matplotlib.pyplot as plt
#%matplotlib inline
from ctapipe.calib import CameraCalibrator
from ctapipe.io.hessio import hessio_event_source
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.image.hillas import hillas_parameters, HillasParameterizationError
from ctapipe.instrument import CameraGeometry
from ctapipe.reco.HillasReconstructor import HillasReconstructor
from ctapipe.reco.shower_max import ShowerMaxEstimator
from ctapipe.image.muon.muon_reco_functions import analyze_muon_event
from ctapipe.coordinates import CameraFrame, NominalFrame, HorizonFrame, GroundFrame, TiltedGroundFrame, TelescopeFrame
from IPython import embed

muon_shower_impacts = []
muon_shower_angles = []
muon_prod_heights = []
muon_prod_depths = []
muon_shower_depths = []
mc_true_energy = []

#filename = '/Users/amitchell/gamma_20deg_180deg_run1000___cta-prod3-demo_desert-2150m-Paranal-demo2rad_cone10.simtel.gz'
#filename = '/Volumes/CTA_USB/CTAMuonData/proton_20deg_180deg_run1298___cta-prod3-merged_desert-2150m-Paranal-subarray-3.simtel-dst0.gz'


def processfile(filename):
    source = hessio_event_source(filename,max_events=None)
    cal = CameraCalibrator(None,None)
    fit = HillasReconstructor()

    cam_geom = {}
    tel_phi = {}
    tel_theta = {}

    for event in source:
        cal.calibrate(event)
        hillas_dict = {}
    
        muon_evt = analyze_muon_event(event)
        mutel = []
    
        if muon_evt['MuonIntensityParams']:
            nmu = 0
            n = 0
            for mip in muon_evt['MuonIntensityParams']:
                if mip:
                    #print("mip=",mip)
                    mutel.append(muon_evt['TelIds'][n])
                    nmu += 1
                n += 1
    
        #Only both reconstructing the relevant showers
        if len(mutel) < 1:
            continue
        else:
            print("Event contained",nmu,"muons and",len(event.dl0.tels_with_data),"tels")
    
        estim = ShowerMaxEstimator(atmosphere_profile_name='paranal')
        #print("MC parameters E=",event.mc.energy,"h=",event.mc.h_first_int)
        en =event.mc.energy.to(u.TeV)
        h_first_int = event.mc.h_first_int.to(u.km)
        az = 70.*u.deg
        for tel_id in event.dl0.tels_with_data:
            if tel_id in mutel:
                print("Tel",tel_id,"contains a muon")
                continue
            if tel_id not in cam_geom:
                cam_geom[tel_id] = CameraGeometry.guess(
                    event.inst.pixel_pos[tel_id][0],
                    event.inst.pixel_pos[tel_id][1],
                    event.inst.optical_foclen[tel_id])

                tel_phi[tel_id] = event.mc.tel[tel_id].azimuth_raw * u.rad
                tel_theta[tel_id] = (np.pi / 2 - event.mc.tel[tel_id].altitude_raw) * u.rad

                pmt_signal = event.r0.tel[tel_id].adc_sums[0]

                mask = tailcuts_clean(cam_geom[tel_id], pmt_signal,
                                      picture_thresh=10., boundary_thresh=5.)
                pmt_signal[mask == 0] = 0

                try:
                    moments = hillas_parameters(event.inst.pixel_pos[tel_id][0],
                                                event.inst.pixel_pos[tel_id][1],
                                                pmt_signal)
                    hillas_dict[tel_id] = moments
                except HillasParameterizationError as e:
                    #print(e)
                    continue

        if len(hillas_dict) < 2: 
            continue

        fit_result = fit.predict(hillas_dict, event.inst, tel_phi, tel_theta)

        #print(fit_result)
        fit_result.alt.to(u.deg)
        fit_result.az.to(u.deg)
        fit_result.core_x.to(u.m)
        az = fit_result.az
        alt = fit_result.alt
        print("Fit result for tels:",fit_result.tel_ids,"Fitted altitude = ",fit_result.alt,"azimuth = ",az,"core position",fit_result.core_x,fit_result.core_y)
        fit_result.core_x *= -1.
        fit_result.core_y *= -1.
        #assert fit_result.is_valid
        #return
    
        #Assume a certain mc direction (altitude, not zenith)
        mcaltaz = HorizonFrame(alt=70.*u.deg,az=180.*u.deg)
        
        #Get muon position and calculate distance to core
        #if mutel: (loop over all muons; don't assume one only
        for muev in mutel:
            tmx = muon_evt['TelIds'].index(muev)
            #print("event.inst.tel_pos[tmx]",event.inst.tel_pos[muev])
            #telpos in ground frame - put into tilted?
            telpos = GroundFrame(x=event.inst.tel_pos[muev][0],y=event.inst.tel_pos[muev][1],z=event.inst.tel_pos[muev][2])
            tilttelpos = telpos.transform_to(TiltedGroundFrame(pointing_direction=mcaltaz))
            #print("tilttelpos=",tilttelpos.x,tilttelpos.y)
            mux = muon_evt['MuonIntensityParams'][tmx].impact_parameter_pos_x + tilttelpos.x
            muy = muon_evt['MuonIntensityParams'][tmx].impact_parameter_pos_y + tilttelpos.y
            #This is in telescope and ground - transform to tilted frame
            print("MuonPos = ",mux,muy,"telpos = ",tilttelpos.x,tilttelpos.y,"showerpos = ",fit_result.core_x,fit_result.core_y)
            muonshowerimpact = np.sqrt((mux-fit_result.core_x)**2. + (muy-fit_result.core_y)**2.)
            print("Found muonshowerimpact = ",muonshowerimpact)
            muon_shower_impacts.append(muonshowerimpact)
                    
            mupos = NominalFrame(x=muon_evt['MuonRingParams'][tmx].ring_center_x,
                                 y=muon_evt['MuonRingParams'][tmx].ring_center_y,
                                 array_direction=mcaltaz,
                                 pointing_direction=mcaltaz)

            #Need to give an altaz??
            mualtaz = mupos.transform_to(HorizonFrame(pointing_direction=mcaltaz))
            #Which frame are the fit results in? (Assume telescope)
            
            #fitcoord = TelescopeFrame(x=fit_result.alt,y=fit_result.az,pointing_direction=mcaltaz)
            #focal_length=event.inst.optical_foclen[mutel[0]])
            #recoaltaz = fitcoord.transform_to(HorizonFrame())
            recoaltaz = HorizonFrame(alt=fit_result.alt,az=fit_result.az)
            #print("Coordinates: mupos",mupos.x,mupos.y)
            #print("mualtaz",mualtaz.alt.value,mualtaz.az.value)
            #print("recoaltaz",recoaltaz.alt.value,recoaltaz.az.value)
            #print("mcaltaz",mcaltaz.alt.value,mcaltaz.az.value)
        
            mush_ang = mualtaz.separation(recoaltaz)
            print("Found muon shower angle of:",mush_ang.to(u.degree))
            muon_shower_angles.append(mush_ang.to(u.degree))
        
            mpheight = muonshowerimpact / np.tan(mush_ang.to(u.rad))
            print("Found muon production height of ",mpheight)
            muon_prod_heights.append(mpheight)

            mpdepth = estim.thickness_profile(mpheight)
            print("Muon Production depth = ",mpdepth)
            muon_prod_depths.append(mpdepth)
            
            #muon shower axis distance z: (or rather, distance along shower axis?)
            zd = muonshowerimpact / np.sin(mush_ang.to(u.rad))
            print("Tentative z = ",zd)
            
        fit_result.h_max = estim.find_shower_max_height(en,h_first_int,alt)#az)
        print("Found h_max = ",fit_result.h_max,"for first int height=",h_first_int,"and mc E = ",en)
        #N.B. this is the height --> want value of Xmax (& Xmumax)
        #embed()
        i = 0
        while i < len(mutel):
            muon_shower_depths.append(fit_result.h_max.value)
            mc_true_energy.append(en)
            i += 1
        
    return


def plot_distributions():
    
    #Diagnostic plots (move to separate script?)
    figi, axi = plt.subplots(1,1,figsize=(10,10))
    figa, axa = plt.subplots(1,1,figsize=(10,10))
    figw, axw = plt.subplots(1,1,figsize=(10,10))
    figpd, axpd = plt.subplots(1,1,figsize=(10,10))
    figsd, axsd = plt.subplots(1,1,figsize=(10,10))
    nbins = 20

    t = Table.read("muonshowertable.fits")
    
    xi = np.linspace(min(t['MuonShowerImpact']),max(t['MuonShowerImpact']),nbins)
    histi = axi.hist(t['MuonShowerImpact'],nbins)
    axi.set_xlim(0.2*min(t['MuonShowerImpact']),1.2*max(t['MuonShowerImpact']))
    axi.set_ylim(1.2*max(histi[0]))
    axi.set_xlabel('Muon Shower Impact (m)')
    #embed()
    #plt.draw()
    figi.savefig("MuonShowerImpactTest.png")

    #plt.hist(msalist)
    xa = np.linspace(min(t['MuonShowerAngle']),max(t['MuonShowerAngle']),nbins)
    hista = axa.hist(t['MuonShowerAngle'],nbins)
    axa.set_xlim(0.2*min(t['MuonShowerAngle']),1.2*max(t['MuonShowerAngle']))
    axa.set_ylim(1.2*max(hista[0]))
    axa.set_xlabel('Muon Shower Angle (^{o})')
    #plt.draw()
    figa.savefig("MuonShowerAngleTest.png")

    xw = np.linspace(min(t['MuonHeight']),max(t['MuonHeight']),nbins)
    histw = axw.hist(t['MuonHeight'],nbins)
    axw.set_xlim(0.2*min(t['MuonHeight']),1.2*max(t['MuonHeight']))
    axw.set_ylim(1.2*max(histw[0]))
    axw.set_xlabel('Muon Production Height')
    #plt.draw()
    figw.savefig("MuonProductionHeightTest.png")


    xpd = np.linspace(min(t['MuonProdDepth']),max(t['MuonProdDepth']),nbins)
    histpd = axpd.hist(t['MuonProdDepth'],nbins)
    axpd.set_xlim(0.2*min(t['MuonProdDepth']),1.2*max(t['MuonProdDepth']))
    axpd.set_ylim(1.2*max(histpd[0]))
    axpd.set_xlabel('Muon Production Depth')
    #plt.draw()
    figpd.savefig("MuonProductionDepthTest.png")

    xsd = np.linspace(min(t['MuonShowerDepth']),max(t['MuonShowerDepth']),nbins)
    histsd = axsd.hist(t['MuonShowerDepth'],nbins)
    axsd.set_xlim(0.2*min(t['MuonShowerDepth']),1.2*max(t['MuonShowerDepth']))
    axsd.set_ylim(1.2*max(histsd[0]))
    axsd.set_xlabel('Muon Shower Depth')
    #plt.draw()
    figsd.savefig("MuonShowerDepthTest.png")

    return
    
    
if __name__=='__main__':
    #Write lists to fits for each run, then combine tables (?)

    filelist = open('/Users/alisonmitchell/protonsimlist.dat','r')
    for line in filelist:
        filename = line.split()
        print("Processing file",filename[0])
        processfile(filename[0])
        pyhessio.close_file()
        #Need to close file (only done in ctapipe if maxevents set)
    
    msilist = []
    for msi in muon_shower_impacts:
        msilist.append(msi.value)


    msalist = []
    for msa in muon_shower_angles:
        #print(msa[0].value)
        msalist.append(msa[0].value)


    mphlist = []
    for mph in muon_prod_heights:
        mphlist.append(mph[0].value)
    #plt.hist(mphlist)


    #Loop over muon & shower depths --> also add to table
    mpdlist = []
    for mpd in muon_prod_depths:
        mpdlist.append(mpd[0].value)

    msdlist = []
    for msd in muon_shower_depths:
        #embed()
        msdlist.append(msd)#.value)

    mcelist = []
    for mce in mc_true_energy:
        mcelist.append(mce.value)
        
    t = Table([msilist,msalist,mphlist,mpdlist,msdlist,mcelist],
              names=("MuonShowerImpact","MuonShowerAngle","MuonHeight","MuonProdDepth","MuonShowerDepth","TrueShowerEnergy"))
    t['MuonShowerImpact'].unit = 'm'
    t['MuonShowerAngle'].unit = 'deg'
    t['MuonHeight'].unit = 'm'
    t['MuonProdDepth'].unit = 'g / cm2'
    t['MuonShowerDepth'].unit = 'g / cm2'
    t['TrueMuonEnergy'].unit = 'TeV'
    
    t.write("muonshowertable.fits",overwrite=True)
    
    embed()

    plot_distributions()

    print("Finished")


