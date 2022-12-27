
import React from "react";
import PropTypes from "prop-types";
import Sub from "./Sub";
import GraphModel from "../../../models/Graph";
import GraphVizReader from "../../../formats/GraphVizReader";
import GraphJSONReader from "../../../formats/GraphJSONReader";
import Style from "./Sub.module.css";
import GraphVizWriter from "../../../formats/GraphVizWriter";

export default class SubSimulate extends Sub
{
	constructor( tProps )
	{
		// Inheritance
		super( tProps );

		// State
		this.state.time = "1";
		this.state.system = 0;
		this.state.language= 0;

		// Variables
		this._file = null;
		this._systems =
		[
			"PHYSICAL",
			"BIOLOGICAL"
		];
		this._languages =
		[
			"FORTRAN",
			"PYTHON_LSODA",
			"PYTHON_BDF",
			"PYTHON_Radau",
			"PYTHON_DOP853",
			"PYTHON_RK23",
			"PYTHON_RK45",
			"PYTHON_bdf",
			"PYTHON_rk5",
			"PYTHON_rk8",
			"PYTHON_beuler",
			"PYTHON_trapz",
			"JULIA",
			"JULIA_NUMBA"
		];

		// Events
		this._onTime = ( tEvent ) => {
		    console.log(this.state);
		    this.setState( { time:  tEvent.target.value  } ); };
		this._onSystem = (tEvent) => { this.setState({system: parseInt(tEvent.target.value)}); }
		this._onLanguage = (tEvent) => { this.setState({language: parseInt(tEvent.target.value)}); }
	    this._onRun = () => { this.onRun(); };
	}

	onRun()
	{
		console.log("onRun!")
		const writer = new GraphVizWriter();
		var selected_string = ""

		for (var selected_node in this.props.graph._selectedNodes){
		    console.log(selected_node)
            selected_string=selected_string+selected_node+","
        }

		window.test({
		    graph: JSON.stringify( writer.write( this.props.graph ) ),
		    selected: selected_string,
		    state: JSON.stringify(this.state)});
	}


	renderContent( tStyle = Style )
	{
		return (
			<React.Fragment>
				<div className={ tStyle.kvp }>
					<span>Time</span>
					<input value={this.state.time} onChange={ this._onTime }/>
					<span>System</span>
					<select onChange={ this._onSystem }>
						{
							this._systems.map(
								( tName, tIndex ) =>
								(
									<option key={ tName } value={ tIndex }>{ tName }</option>
								)
							)
						}
					</select>
					<span>Language</span>
					<select onChange={ this._onLanguage }>
						{
							this._languages.map(
								( tName, tIndex ) =>
								(
									<option key={ tName } value={ tIndex }>{ tName }</option>
								)
							)
						}
					</select>
				</div>
				<div className={ tStyle.buttons }>
					<button className={ tStyle.button } onClick={ this._onRun }>run</button>
				</div>
			</React.Fragment>
		);
	}
}

SubSimulate.propTypes = Object.assign(
	{
		graph: PropTypes.instanceOf( GraphModel ).isRequired
	},
	Sub.propTypes
);

SubSimulate.defaultProps =
{
	title: "run"
};