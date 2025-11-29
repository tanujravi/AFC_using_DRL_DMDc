/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2023 
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    svdToFoam

Description

    Fetch SVD reconstruction and left singular vectors from SmartRedis.

\*---------------------------------------------------------------------------*/

#include "error.H"
#include "fvCFD.H"
#include "wordList.H"
#include "timeSelector.H"
#include "client.h"
#include <iomanip>
#include <vector>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    // Selecting the time step folders for the approximated fields.
    timeSelector::addOptions();

    // Add the option to the application for the name of the approximated field.
    argList::addOption
    (
        "fieldName", 
        "fieldName",
        "Name of the approximated field, e.g. p." 
    );

    argList::addOption
    (
        "svdRank", 
        "svdRank",
        "SVD rank used for the reconstruction." 
    );

    argList::addOption
    (
        "FOName", 
        "FOName",
        "Name of the SmartRedis function object used to write the data." 
    ); 

    // OpenFOAM boilerplate: set root folder and options
    #include "setRootCase.H"
    #include "createTime.H"
    //runTime.setTime(0.0, 0);
    
    // Read the name of the approximated field from the command line option
    // foamSmartSimDmd -case cavity -fieldName p
    #include "createMesh.H"
    const word fieldName = args.get<word>("fieldName");
    const word svdRank = args.get<word>("svdRank");
    const word FOName = args.get<word>("FOName");
    Info << "Transferring reconstruction with rank " << svdRank
         << " of field " << fieldName
         << " originally written by function object " << FOName << endl;

    SmartRedis::Client client(false);

    const auto mpiRank = std::to_string(Pstream::myProcNo());
    auto baseName = "rec_ensemble_r" + svdRank + "_field_name_" + fieldName + "_" + mpiRank + 
        ".rank_" + svdRank + "_field_name_" + fieldName +
        "_mpi_rank_" + mpiRank;

    auto endTime = runTime.endTime().value();
    auto deltaT = runTime.deltaT().value();
    Info << "current time" << runTime.value() << endl;

    bool isRunning = true;
    while (isRunning)
    {
        
        if (runTime.writeTime())
        {
            Info << "Writing reconstruction at time " << runTime.value() << endl;
            auto tensorName = baseName + "_time_index_" + std::to_string(runTime.timeIndex());
            Info << "tensorName " << tensorName << endl;
            
            auto tensorInDB = client.tensor_exists(tensorName);
            if (tensorInDB)
            {

                if (fieldName == "U") {
                Info << "fieldName " << fieldName << endl;

                volVectorField Utemplate
                (
                    IOobject
                    (
                        "U",                       // template field present in the case
                        "0",
                        mesh,
                        IOobject::MUST_READ,       // ensure we get BCs
                        IOobject::NO_WRITE
                    ),
                    mesh
                );
                volVectorField U
                (
                    IOobject
                    (
                        //"recU_r_" + svdRank,
                        "U",
                        runTime.timeName(),
                        mesh,
                        IOobject::NO_READ,
                        IOobject::AUTO_WRITE
                    ),
                    Utemplate
                );
                

                SRTensorType get_type;
                std::vector<size_t> get_dims;
                void* reconstruction;
                client.get_tensor(tensorName, reconstruction, get_dims, get_type, SRMemLayoutNested);

                forAll (U.internalFieldRef(), cellI)
                {
                    auto Ui = vector(
                        ((double**)reconstruction)[cellI][0], ((double**)reconstruction)[cellI][1], ((double**)reconstruction)[cellI][2]
                    );
                    U.internalFieldRef()[cellI] = Ui;
                }
                U.correctBoundaryConditions();
                U.write();
            }

                else {
                    Info << "fieldName  else" << fieldName << endl;

                    volScalarField scalartemplate
                    (
                        IOobject
                        (
                            fieldName,                       // template field present in the case
                            "0",
                            mesh,
                            IOobject::MUST_READ,       // ensure we get BCs
                            IOobject::NO_WRITE
                        ),
                        mesh
                    );
                    volScalarField scalar
                    (
                        IOobject
                        (
                            //"rec_" + fieldName + "_r_" + svdRank,
                            fieldName,
                            runTime.timeName(),
                            mesh,
                            IOobject::NO_READ,
                            IOobject::AUTO_WRITE
                        ),
                        scalartemplate
                    );
                SRTensorType get_type;
                std::vector<size_t> get_dims;
                void* reconstruction;
                client.get_tensor(tensorName, reconstruction, get_dims, get_type, SRMemLayoutNested);
                forAll (scalar.internalFieldRef(), cellI)
                {   
                    scalar.internalFieldRef()[cellI] = ((double*)reconstruction)[cellI];
                }
                scalar.correctBoundaryConditions();
                scalar.write();
                    
                }


        }
        }
        isRunning = runTime.value() < (endTime - 0.5*deltaT);
        if (isRunning)
        {
            runTime++;
        }
        

    if (!isRunning)
    {
        auto modeBaseName =
            "rec_ensemble_r" + svdRank +
            "_field_name_" + fieldName + "_" + mpiRank +
            ".global_U_field_name_" + fieldName +
            "_mpi_rank_" + mpiRank;

        // --- Vector field case: U ---
        if (fieldName == "U")
        {
            for (int r = 0; r < std::stoi(svdRank); r++)
            {
                auto tensorName = modeBaseName + "_mode_" + std::to_string(r);
                auto tensorInDB = client.tensor_exists(tensorName);

                if (tensorInDB)
                {
                    Info << "Writing POD mode " << std::to_string(r) << endl;

                    volVectorField modeU
                    (
                        IOobject
                        (
                            fieldName + "_mode_" + std::to_string(r+1),
                            runTime.timeName(),
                            mesh,
                            IOobject::NO_READ,
                            IOobject::AUTO_WRITE
                        ),
                        mesh,
                        dimensionedVector
                        (
                            "m",           // arbitrary name/label
                            dimVelocity,   // dimensions: velocity
                            vector::zero
                        )
                    );

                    SRTensorType get_type;
                    std::vector<size_t> get_dims;
                    void* mode;
                    client.get_tensor(tensorName, mode, get_dims, get_type, SRMemLayoutNested);

                    forAll (modeU.internalFieldRef(), cellI)
                    {
                        auto Ui = vector(
                            ((double**)mode)[cellI][0],
                            ((double**)mode)[cellI][1],
                            ((double**)mode)[cellI][2]
                        );
                        modeU.internalFieldRef()[cellI] = Ui;
                    }
                    modeU.write();
                }
            }
        }
        // --- Scalar field case: p, T, etc. ---
        else
        {
            // Template field with correct dimensions + BCs, read once from time "0"
            volScalarField scalarTemplate
            (
                IOobject
                (
                    fieldName,
                    "0",
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE
                ),
                mesh
            );

            for (int r = 0; r < std::stoi(svdRank); r++)
            {
                auto tensorName = modeBaseName + "_mode_" + std::to_string(r);
                auto tensorInDB = client.tensor_exists(tensorName);

                if (tensorInDB)
                {
                    Info << "Writing POD mode " << std::to_string(r) << endl;

                    // Use template to inherit dimensions & BCs
                    volScalarField modeScalar
                    (
                        IOobject
                        (
                            fieldName + "_mode_" + std::to_string(r+1),
                            runTime.timeName(),
                            mesh,
                            IOobject::NO_READ,
                            IOobject::AUTO_WRITE
                        ),
                        scalarTemplate
                    );

                    SRTensorType get_type;
                    std::vector<size_t> get_dims;
                    void* mode;
                    client.get_tensor(tensorName, mode, get_dims, get_type, SRMemLayoutNested);

                    forAll (modeScalar.internalFieldRef(), cellI)
                    {
                        modeScalar.internalFieldRef()[cellI] =
                            ((double*)mode)[cellI];
                    }
                    modeScalar.write();
                }
            }
        }
    }    
    }

    return 0;
}


// ************************************************************************* //
