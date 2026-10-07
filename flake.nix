{
  description = "2D/3D finite element programming framework";

  inputs = {
    nixpkgs.url = "https://channels.nixos.org/nixpkgs-unstable/nixexprs.tar.zst";
  };

  inputs.self.submodules = true;

  outputs = inputs: {
    packages = builtins.mapAttrs (
      system: pkgs:
      let
        pname = "nla3d";
        version = "0.0.0";
        src = ./.;
        python = pkgs.python3;
      in
      {
        nla3d = pkgs.stdenv.mkDerivation {
          inherit pname version src;
          nativeBuildInputs = [ pkgs.cmake ];

          cmakeFlags = [
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
          ];
        };

        nla3d_py = python.pkgs.buildPythonPackage {
          pname = "nla3d";
          inherit version src;

          nativeBuildInputs = [
            pkgs.cmake
            pkgs.swig
          ];
          buildInputs = [ python ];
          propagatedBuildInputs = [ python ];

          pyproject = false;

          cmakeFlags = [
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
            "-DNLA3D_PYTHON=ON"
          ];

          buildPhase = ''
            runHook preBuild
            cmake --build . --target nla3d_py
            runHook postBuild
          '';

          installPhase =
            let
              site-packages = python.sitePackages;
              extension_suffix = builtins.readFile (
                pkgs.runCommandNoCC "python-extension-suffix" {
                  nativeBuildInputs = [ python ];
                } ''
                  python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"), end="")' > $out
                ''
              );
            in
            ''
              runHook preInstall

              mkdir -p $out/${site-packages}

              cp python/_nla3d${extension_suffix} python/nla3d.py $out/${site-packages}/

              runHook postInstall
            '';
        };

        nla3d-with-mkl = pkgs.stdenv.mkDerivation {
          inherit pname version src;
          nativeBuildInputs = [
            pkgs.cmake
          ];
          buildInputs = [ pkgs.mkl ];

          cmakeFlags = [
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
            "-DNLA3D_USE_MKL=ON"
          ];
        };

        default = inputs.self.packages.${system}.nla3d;
      }
    ) inputs.nixpkgs.legacyPackages;

    devShells = builtins.mapAttrs (system: pkgs: {
      default = pkgs.mkShellNoCC {
        packages = with pkgs; [
          llvmPackages.clang
          llvmPackages.openmp
          gdb
          cmake
          swig
          (python3.withPackages (p: [ inputs.self.packages.${system}.nla3d_py ]))
        ];
      };
    }) inputs.nixpkgs.legacyPackages;
  };
}
