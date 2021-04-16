use log::{info};

pub fn write_header() {
    info!("{: ^80}", "-----------------");
    info!("{: ^80}", "TINCR");
    info!("{: ^80}", "-----------------");
    let mut version_string: String = "version: ".to_owned();
    version_string.push_str(crate_version!());
    info!("{: ^80}", version_string);
    info!("{: ^80}", "");
    info!("{: ^80}", "::::::::::::::::::::::::::::::::::::::");
    info!("{: ^80}", "::           Roland Mitric          ::");
    info!("{: ^80}", "::  Chair of theoretical chemistry  ::");
    info!("{: ^80}", "::      University of Wuerzburg     ::");
    info!("{: ^80}", "::::::::::::::::::::::::::::::::::::::");
    // info!("{: ^80}", ":::::::::::::::::::::::::::::::::::::::::::::::::::::::::");
    // info!("{: ^80}", ":: Authors:                                            ::");
    // info!("{: ^80}", ":: Joscha Hoche <joscha.hoche@uni-wuerzburg.de>        ::");
    // info!("{: ^80}", ":: Richard Einsele <richard.einsele@uni-wuerzburg.de>  ::");
    // info!("{: ^80}", ":: Richard Einsele <richard.einsele@uni-wuerzburg.de>  ::");
    // info!("{: ^80}", ":: Xincheng Miao <xincheng.miao@uni-wuerzburg.de>      ::");
    // info!("{: ^80}", ":::::::::::::::::::::::::::::::::::::::::::::::::::::::::");
    info!("{: ^80}", "");
}