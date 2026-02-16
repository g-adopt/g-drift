#!/usr/bin/env python3
"""Apply metadata enrichment for the 22 new GRD-collection seismic models."""

import json
from pathlib import Path

NEW_MODEL_METADATA = {
    "3d_seismic_3D2015-07Sv": {
        "source": "Debayle, E., Dubuffet, F., & Durand, S. (2016). An automatically updated S-wave model of the upper mantle and the depth extent of azimuthal anisotropy. Geophysical Research Letters, 43(2), 674-682.",
        "doi": "10.1002/2015GL067329",
        "year": 2016,
        "description": "Global Sv-wave velocity model of the upper mantle from waveform modelling of Rayleigh waves.",
    },
    "3d_seismic_AF2019": {
        "source": "Celli, N. L., Lebedev, S., Schaeffer, A. J., & Gaina, C. (2020). African cratonic lithosphere carved by mantle plumes. Nature Communications, 11, 92.",
        "doi": "10.1038/s41467-019-13871-2",
        "year": 2020,
        "description": "Regional S-wave velocity model of the African upper mantle and transition zone from waveform tomography.",
    },
    "3d_seismic_ANT-20": {
        "source": "Lloyd, A. J., Wiens, D. A., Zhu, H., Tromp, J., Nyblade, A. A., Aster, R. C., Hansen, S. E., Dalziel, I. W. D., Wilson, T. J., Ivins, E. R., & O'Donnell, J. P. (2020). Seismic structure of the Antarctic upper mantle imaged with adjoint tomography. Journal of Geophysical Research: Solid Earth, 125(3), e2019JB017823.",
        "doi": "10.1029/2019JB017823",
        "year": 2020,
        "description": "Regional 3D seismic model of the Antarctic upper mantle from adjoint (full-waveform) tomography.",
    },
    "3d_seismic_AuSREM": {
        "source": "Kennett, B. L. N., & Salmon, M. (2012). AuSREM: Australian Seismological Reference Model. Australian Journal of Earth Sciences, 59(8), 1091-1103.",
        "doi": "10.1080/08120099.2012.736406",
        "year": 2012,
        "description": "Regional seismological reference model for the Australian continent providing P- and S-wave velocity structure.",
    },
    "3d_seismic_Aus22": {
        "source": "de Laat, J. I., Lebedev, S., Celli, N. L., Bonadio, R., Chagas de Melo, B., & Rawlinson, N. (2023). Structure and evolution of the Australian plate and underlying upper mantle from waveform tomography with massive data sets. Geophysical Journal International, 234(1), 153-189.",
        "doi": "10.1093/gji/ggad062",
        "year": 2023,
        "description": "Regional S-wave velocity model of the upper mantle beneath the Australian plate from waveform inversion.",
    },
    "3d_seismic_CAM2016": {
        "source": "Priestley, K., McKenzie, D., & Ho, T. (2018). A Lithosphere-Asthenosphere Boundary - a Global Model Derived from Multimode Surface-Wave Tomography and Petrology. In Lithospheric Discontinuities, AGU Geophysical Monograph Series.",
        "doi": "10.1002/9781119249740.ch6",
        "year": 2018,
        "description": "Global upper mantle Vsv model from waveform modelling of Rayleigh waveforms.",
    },
    "3d_seismic_CSEM-2019": {
        "source": "Fichtner, A., van Herwaarden, D.-P., Afanasiev, M., Simute, S., Krischer, L., Cubuk-Sabuncu, Y., Taymaz, T., Colli, L., Saygin, E., Villasenor, A., Trampert, J., Cupillard, P., Bunge, H.-P., & Igel, H. (2018). The Collaborative Seismic Earth Model: Generation 1. Geophysical Research Letters, 45(9), 4007-4016.",
        "doi": "10.1029/2018GL077338",
        "year": 2018,
        "description": "Global collaborative seismic Earth model integrating full-waveform inversion results from regional studies.",
    },
    "3d_seismic_DETOX-P02": {
        "source": "Hosseini, K., Sigloch, K., Tsekhmistrenko, M., Zaheri, A., Nissen-Meyer, T., & Igel, H. (2020). Global mantle structure from multifrequency tomography using P, PP and P-diffracted waves. Geophysical Journal International, 220(1), 96-141.",
        "doi": "10.1093/gji/ggz394",
        "year": 2020,
        "description": "Global P-wave tomography model of the whole mantle using multifrequency body-wave data.",
    },
    "3d_seismic_F2010-Afr": {
        "source": "Fishwick, S. (2010). Surface wave tomography: Imaging of the lithosphere-asthenosphere boundary beneath central and southern Africa? Lithos, 120(1-2), 63-73.",
        "doi": "10.1016/j.lithos.2010.05.011",
        "year": 2010,
        "description": "Regional surface-wave tomography model of the upper mantle beneath central and southern Africa.",
    },
    "3d_seismic_FR12": {
        "source": "Fishwick, S., & Rawlinson, N. (2012). 3-D structure of the Australian lithosphere from evolving seismic datasets. Australian Journal of Earth Sciences, 59(6), 809-826.",
        "doi": "10.1080/08120099.2012.702319",
        "year": 2012,
        "description": "Regional surface-wave and body-wave tomography model of the Australian lithosphere.",
    },
    "3d_seismic_LLNL-G3D-JPS": {
        "source": "Simmons, N. A., Myers, S. C., Johannesson, G., Matzel, E., & Grand, S. P. (2015). Evidence for long-lived subduction of an ancient tectonic plate beneath the southern Indian Ocean. Geophysical Research Letters, 42, 9270-9278.",
        "doi": "10.1002/2015GL066237",
        "year": 2015,
        "description": "Global joint P- and S-wave tomography model of the whole mantle from LLNL.",
    },
    "3d_seismic_MITS-18": {
        "source": "Golos, E. M., Fang, H., Yao, H., Zhang, H., Burdick, S., Vernon, F., Schaeffer, A., Lebedev, S., & van der Hilst, R. D. (2018). Shear wave tomography beneath the United States using a joint inversion of surface and body waves. Journal of Geophysical Research: Solid Earth, 123, 5169-5189.",
        "doi": "10.1029/2017JB014894",
        "year": 2018,
        "description": "Regional shear-wave velocity model of the crust and upper mantle beneath the contiguous United States.",
    },
    "3d_seismic_PM13": {
        "source": "Moulik, P., & Ekstrom, G. (2014). An anisotropic shear velocity model of the Earth's mantle using normal modes, body waves, surface waves and long-period waveforms. Geophysical Journal International, 199(3), 1713-1738.",
        "doi": "10.1093/gji/ggu356",
        "year": 2014,
        "description": "Global anisotropic shear-wave velocity model of the mantle constrained by normal modes, body waves, and surface waves.",
    },
    "3d_seismic_PMEAN": {
        "source": "Becker, T. W., & Boschi, L. (2002). A comparison of tomographic and geodynamic mantle models. Geochemistry, Geophysics, Geosystems, 3(1), 1003.",
        "doi": "10.1029/2001GC000168",
        "year": 2002,
        "description": "Composite global mean P-wave mantle tomography model constructed by averaging existing tomographic models.",
    },
    "3d_seismic_SA2019": {
        "source": "Celli, N. L., Lebedev, S., Schaeffer, A. J., Ravenna, M., & Gaina, C. (2020). The upper mantle beneath the South Atlantic Ocean, South America and Africa from waveform tomography with massive data sets. Geophysical Journal International, 221(1), 178-204.",
        "doi": "10.1093/gji/ggz574",
        "year": 2020,
        "description": "Regional waveform tomography model of the upper mantle beneath the South Atlantic, South America, and Africa.",
    },
    "3d_seismic_SAVANI": {
        "source": "Auer, L., Boschi, L., Becker, T. W., Nissen-Meyer, T., & Giardini, D. (2014). Savani: A variable resolution whole-mantle model of anisotropic shear velocity variations based on multiple data sets. Journal of Geophysical Research: Solid Earth, 119, 3006-3034.",
        "doi": "10.1002/2013JB010773",
        "year": 2014,
        "description": "Global whole-mantle radially anisotropic shear-wave velocity model with variable block parameterization.",
    },
    "3d_seismic_SEMUM2": {
        "source": "French, S. W., Lekic, V., & Romanowicz, B. (2013). Waveform tomography reveals channeled flow at the base of the oceanic asthenosphere. Science, 342(6155), 227-230.",
        "doi": "10.1126/science.1241514",
        "year": 2013,
        "description": "Global radially anisotropic shear-velocity model of the upper mantle from spectral-element waveform tomography.",
    },
    "3d_seismic_SL2013NA": {
        "source": "Schaeffer, A. J., & Lebedev, S. (2014). Imaging the North American continent using waveform inversion of global and USArray data. Earth and Planetary Science Letters, 402, 26-41.",
        "doi": "10.1016/j.epsl.2014.06.014",
        "year": 2014,
        "description": "Upper-mantle shear-wave speed model focused on North America from multimode waveform inversion.",
    },
    "3d_seismic_SL2013sv": {
        "source": "Schaeffer, A. J., & Lebedev, S. (2013). Global shear speed structure of the upper mantle and transition zone. Geophysical Journal International, 194(1), 417-449.",
        "doi": "10.1093/gji/ggt095",
        "year": 2013,
        "description": "Global Vsv model of the upper mantle and transition zone from automated multimode waveform inversion.",
    },
    "3d_seismic_SL2013sv-uninterp": {
        "source": "Schaeffer, A. J., & Lebedev, S. (2013). Global shear speed structure of the upper mantle and transition zone. Geophysical Journal International, 194(1), 417-449.",
        "doi": "10.1093/gji/ggt095",
        "year": 2013,
        "description": "Uninterpolated (raw parameterization) version of the SL2013sv global shear-wave speed model.",
    },
    "3d_seismic_SMEAN": {
        "source": "Becker, T. W., & Boschi, L. (2002). A comparison of tomographic and geodynamic mantle models. Geochemistry, Geophysics, Geosystems, 3(1), 1003.",
        "doi": "10.1029/2001GC000168",
        "year": 2002,
        "description": "Composite global mean S-wave velocity model of the whole mantle averaging three existing tomographic models.",
    },
    "3d_seismic_Y14": {
        "source": "Yuan, H., French, S., Cupillard, P., & Romanowicz, B. (2014). Lithospheric expression of geological units in central and eastern North America from full waveform tomography. Earth and Planetary Science Letters, 402, 176-186.",
        "doi": "10.1016/j.epsl.2013.11.057",
        "year": 2014,
        "description": "Regional radially anisotropic shear-wave velocity model of the North American upper mantle from full waveform tomography.",
    },
}


def main():
    manifest_path = Path(__file__).parent.parent / "gdrift" / "datasets.json"

    with open(manifest_path) as f:
        manifest = json.load(f)

    updated = 0
    for dataset in manifest["datasets"]:
        name = dataset["name"]
        if name in NEW_MODEL_METADATA:
            meta = NEW_MODEL_METADATA[name]
            dataset["source"] = meta["source"]
            dataset["doi"] = meta["doi"]
            dataset["year"] = meta["year"]
            dataset["description"] = meta["description"]
            updated += 1
            print(f"  Updated {name}")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"\nUpdated {updated}/{len(NEW_MODEL_METADATA)} models in {manifest_path}")

    # Verify no UNKNOWN sources remain
    unknown = [d["name"] for d in manifest["datasets"]
               if "UNKNOWN" in d.get("source", "")]
    if unknown:
        print(f"\nWARNING: {len(unknown)} models still have UNKNOWN source: {unknown}")
    else:
        print("\nAll models now have proper citations.")


if __name__ == "__main__":
    main()
