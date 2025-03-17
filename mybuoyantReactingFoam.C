/*---------------------------------------------------------------------------*\\
  Application
    ammoniaDiffusionFoam

  Description
    Transient solver for turbulent flow and diffusion of ammonia in air.
    This solver is a simplified version of buoyantReactingFoam,
    removing combustion, reaction, and compressible flow components.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "pimpleControl.H"
#include "CorrectPhi.H"
#include "fvModels.H"
#include "fvConstraints.H"

// Create mesh and time controls
Foam::argList args(argc, argv);
Foam::timeControl runTime(args);
Foam::fvMesh mesh(Foam::IOobject("mesh", runTime.timeName(), Foam::rootPath(), Foam::IOobject::MUST_READ, Foam::IOobject::AUTO_WRITE));

// Velocity and ammonia concentration fields
Foam::volVectorField U(
    Foam::IOobject("U", runTime.timeName(), mesh, Foam::IOobject::MUST_READ, Foam::IOobject::AUTO_WRITE),
    mesh
);

Foam::volScalarField NH3(
    Foam::IOobject("NH3", runTime.timeName(), mesh, Foam::IOobject::MUST_READ, Foam::IOobject::AUTO_WRITE),
    mesh
);

// Time loop
while (runTime.run())
{
    runTime++; // Advance time step

    // Solve velocity equation (ignoring buoyancy and reaction terms)
    Foam::tmp<Foam::fvVectorMatrix> UEqn(
        Foam::fvm::ddt(U) + Foam::fvm::div(phi, U)
    );

    UEqn().relax();
    Foam::solve(UEqn() == -Foam::fvc::grad(NH3));

    // Solve ammonia diffusion equation
    Foam::solve(Foam::fvm::ddt(NH3) + Foam::fvm::div(phi, NH3) - Foam::fvm::laplacian(1e-5, NH3));

    runTime.write(); // Write results
}

Info << "Simulation complete." << endl;

