
import React from "react";
import PropTypes from "prop-types";
import Sub from "./Sub";
import GraphModel from "../../../models/Graph";
import GraphVizReader from "../../../formats/GraphVizReader";
import GraphJSONReader from "../../../formats/GraphJSONReader";
import Style from "./Sub.module.css";

export default class SubSimulate extends Sub
{
	constructor( tProps )
	{
		// Inheritance
		super( tProps );

		// State
		this.state.time = "1";
		this.state.system = 0;
		this.state.mode= 0;

		// Variables
		this._file = null;
		this._systems =
		[
			"PHYSICAL",
			"BIOLOGICAL"
		];
		this._modes =
		[
			"FORTRAN",
			"PYTHON"
		];

		// Events
		this._onTime = ( tEvent ) => {
		    console.log(this.state);
		    this.setState( { time:  tEvent.target.value  } ); };
		this._onSystem = (tEvent) => { this.setState({system: parseInt(tEvent.target.value)}); }
		this._onMode = (tEvent) => { this.setState({mode: parseInt(tEvent.target.value)}); }
	    this._onRun = () => { this.onRun(); };
	}

	onRun()
	{
		console.log("onRun!")
		window.test(this.props);
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
					<select onChange={ this._onMode }>
						{
							this._modes.map(
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