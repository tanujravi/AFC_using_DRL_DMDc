/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) YEAR AUTHOR, AFFILIATION
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

Description
    Template for use with codeStream.

\*---------------------------------------------------------------------------*/

#include "dictionary.H"
#include "Ostream.H"
#include "Pstream.H"
#include "pointField.H"
#include "tensor.H"
#include "unitConversion.H"

//{{{ begin codeInclude
#line 32 "/home/tanuj/Masters_Thesis/AFC_using_DRL_DMDc/dmdc/signal_library/cylinder2D/AM/system/blockMeshDict/#codeStream"
#include "pointField.H"
//}}} end codeInclude

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * Local Functions * * * * * * * * * * * * * * //

//{{{ begin localCode

//}}} end localCode


// * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * //

extern "C" void codeStream_5d1386398f1f60435a3b548a6a6b50e0577d8205(Foam::Ostream& os, const Foam::dictionary& dict)
{
//{{{ begin code
    #line 37 "/home/tanuj/Masters_Thesis/AFC_using_DRL_DMDc/dmdc/signal_library/cylinder2D/AM/system/blockMeshDict/#codeStream"
pointField points({
            /* 0*/ {0, 0, 0},
            /* 1*/ {0.2 * 2, 0, 0},
            /* 2*/ {2.2, 0, 0},
            /* 3*/ {2.2, 0.41, 0},
            /* 4*/ {0.2 * 2, 0.41, 0},
            /* 5*/ {0, 0.41, 0},
            /* 6*/ {0.2 - 0.03535533905932738, 0.2 - 0.03535533905932738, 0},
            /* 7*/ {0.2 + 0.03535533905932738, 0.2 - 0.03535533905932738, 0},
            /* 8*/ {0.2 - 0.03535533905932738, 0.2 + 0.03535533905932738, 0},
            /* 9*/ {0.2 + 0.03535533905932738, 0.2 + 0.03535533905932738, 0}
        });

        // Duplicate z points for thickness
        const label sz = points.size();
        points.resize(2*sz);
        for (label i = 0; i < sz; ++i)
        {
            const point& pt = points[i];
            points[i + sz] = point(pt.x(), pt.y(), 0.01);
        }

        os  << points;
//}}} end code
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //

